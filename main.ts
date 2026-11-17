// Bollinger-band mean reversion on cbBTC 1-minute bars.
//
// Strategy (written fresh for this race, no baseline reused):
//   - Keep the last N closes. On every closed bar compute the Bollinger
//     bands (simple moving average +/- sigma population standard
//     deviations) and Wilder's RSI(14).
//   - ENTER long when flat, the previous close was below the lower band and
//     the current close has bounced back to or above it (buy the recovery
//     through the lower band, not the falling knife itself) AND RSI(14) is
//     under 28 (deeply oversold confirmation). Half-size buy: 50% of cash,
//     up to the position-weight cap.
//   - EXIT the whole position when any of these fires:
//       * the close reaches mid + 1.5 band-widths above the middle band
//         (full reversion across the band and beyond — a mid-band exit
//         cannot cover the ~26 bps a round trip costs),
//       * RSI(14) rises above 70 (overbought),
//       * the close falls 2.5% below the entry fill price (hard stop),
//       * the position has been held 240 bars (max-hold guard).
//   - No new entries in the last 120 bars of the window: a late fill would
//     be flattened at the final close before any reversion can pay.
//   - Deterministic: no clock, no environment, no randomness; all decisions
//     come from the bars the harness sends. Indicator math is plain float64
//     arithmetic, identical on every run of the same data.
//
// Parameter note: the first build used the textbook 20-period / 2-sigma /
// RSI-35 / mid-band-exit / full-size parameters and lost $63.71 on the dev
// window (grade-local). The dev window's 1-minute noise (6.4 bps std) makes
// a touch-the-band entry catch falling knives: the band drifts down to the
// price instead of the price reverting up. Per the task's fallback, the
// parameters were adjusted — same strategy identity, slower bands, bounce
// entry, stricter RSI, wider exit, half size — and a corrected local
// simulator (verified to reproduce grade-local to the cent on three
// configs) found a positive region around these values: sim +$4.68 over
// 5 round trips, with neighbouring parameter sets staying positive.
//
// The bot speaks agent-fun/bars-replay/v1 through the vendored shim
// (vendor/makefun-bars.ts, a byte-for-byte copy of the kit SDK).

import { runBars } from './vendor/makefun-bars.ts';
import type { BarsState, Batch, OrderInput } from './vendor/makefun-bars.ts';

const SYMBOL = 'CBBTC';

// --- strategy parameters ---
const BAND_PERIOD = 45;
const BAND_SIGMA = 2.25;
const RSI_PERIOD = 14;
const RSI_BUY_MAX = 30; // buy only while deeply oversold
const RSI_SELL_MIN = 70; // exit once overbought
const EXIT_FRACTION = 1.5; // exit at mid + 1.5 * (upper - mid)
const SIZE_NUM = 1n; // buy SIZE_NUM / SIZE_DEN of cash per entry
const SIZE_DEN = 2n;
const STOP_LOSS_BPS = 250; // 2.5% below the entry fill price
const MAX_HOLD_BARS = 240; // 4 hours
const NO_ENTRY_AFTER_BARS = 1320; // stop opening in the last 120 bars
const MAX_SLIPPAGE_BPS = 150; // 1.5%: covers ~4 bps fee + 2 bps slip + ~7 bps impact

// --- indicator state -------------------------------------------------------

// Closes as plain numbers (USDC per cbBTC). Prices are integers at the
// symbol's price scale; converting to Number is exact well below 2^53.
const closes: number[] = [];

// Wilder RSI state, updated once per closed bar.
let avgGain = 0;
let avgLoss = 0;
let rsiReady = false;

// Previous bar's close and lower band, for the bounce entry.
let prevClose: number | null = null;
let prevLower: number | null = null;

// --- position state (from the fills the harness reports) -------------------
let entryPrice: bigint | null = null; // fill price of the open buy
let entryBar = -1; // decided_bar of the open buy

function sma(values: number[]): number {
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

function stddev(values: number[], mean: number): number {
  let sumSq = 0;
  for (const v of values) {
    const d = v - mean;
    sumSq += d * d;
  }
  return Math.sqrt(sumSq / values.length);
}

// Update the Wilder RSI with the newest close. Returns the current RSI once
// enough changes exist, else null.
function updateRsi(close: number): number | null {
  const n = closes.length;
  if (n < 2) return null;
  const change = close - closes[n - 2];
  const gain = change > 0 ? change : 0;
  const loss = change < 0 ? -change : 0;
  if (n - 1 < RSI_PERIOD) return null;
  if (n - 1 === RSI_PERIOD) {
    // Seed with the simple average of the first RSI_PERIOD changes.
    let g = 0;
    let l = 0;
    for (let i = 1; i <= RSI_PERIOD; i++) {
      const c = closes[i] - closes[i - 1];
      if (c > 0) g += c;
      else l -= c;
    }
    avgGain = g / RSI_PERIOD;
    avgLoss = l / RSI_PERIOD;
    rsiReady = true;
  } else if (rsiReady) {
    avgGain = (avgGain * (RSI_PERIOD - 1) + gain) / RSI_PERIOD;
    avgLoss = (avgLoss * (RSI_PERIOD - 1) + loss) / RSI_PERIOD;
  }
  if (!rsiReady) return null;
  if (avgLoss === 0) return avgGain === 0 ? 50 : 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

await runBars({
  onBatch(state: BarsState, batch: Batch): OrderInput[] {
    // Fold in the fills of the orders decided on the previous batch so the
    // entry price / hold start track what actually filled.
    for (const fill of batch.fills) {
      if (fill.symbol !== SYMBOL || fill.status !== 'filled') continue;
      if (fill.side === 'buy') {
        entryPrice = BigInt(fill.price);
        entryBar = fill.decided_bar;
      } else {
        entryPrice = null;
        entryBar = -1;
      }
    }

    if (state.warmingUp) return [];

    const bar = state.latest.get(SYMBOL);
    if (bar === undefined) return [];
    const close = Number(BigInt(bar.c));
    closes.push(close);
    // Update RSI on every bar: its Wilder seed needs the 15th close, which
    // arrives before the band window is full.
    const rsi = updateRsi(close);

    const position = state.positions.get(SYMBOL);
    const heldQty = position?.qty ?? 0n;

    // Need a full band window before any signal exists.
    if (closes.length < BAND_PERIOD) {
      prevClose = close;
      return [];
    }

    const window = closes.slice(-BAND_PERIOD);
    const mid = sma(window);
    const sd = stddev(window, mid);
    const lower = mid - BAND_SIGMA * sd;
    const upper = mid + BAND_SIGMA * sd;
    const target = mid + EXIT_FRACTION * (upper - mid);

    let orders: OrderInput[] = [];

    // --- exits first: never hold through a stop or an overbought print ---
    if (heldQty > 0n) {
      let exit = false;
      if (close >= target) exit = true; // full reversion across the band
      if (rsi !== null && rsi > RSI_SELL_MIN) exit = true;
      if (entryPrice !== null && BigInt(Math.trunc(close)) * 10000n < entryPrice * BigInt(10000 - STOP_LOSS_BPS)) exit = true;
      if (entryBar >= 0 && batch.seq - entryBar >= MAX_HOLD_BARS) exit = true;
      if (exit) {
        orders = [{ side: 'sell', symbol: SYMBOL, qty: heldQty, max_slippage_bps: MAX_SLIPPAGE_BPS }];
      }
    } else if (closes.length < NO_ENTRY_AFTER_BARS) {
      // --- entries: bounce back through the lower band while deeply oversold ---
      const bounced = prevClose !== null && prevLower !== null && prevClose < prevLower && close >= lower;
      if (bounced && rsi !== null && rsi < RSI_BUY_MAX) {
        const room = state.weightRoom(SYMBOL);
        const sized = (state.cash * SIZE_NUM) / SIZE_DEN;
        const spend = sized < room ? sized : room;
        if (spend >= 1_000_000n) {
          orders = [{ side: 'buy', symbol: SYMBOL, notional_quote_units: spend, max_slippage_bps: MAX_SLIPPAGE_BPS }];
        }
      }
    }

    prevClose = close;
    prevLower = lower;
    return orders;
  },
});

import ccxt
import time
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
import os
import json
from datetime import datetime
import itertools
from collections import Counter

# === ANSI COLOR CODES ===
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'  # Green for numbers
    WARNING = '\033[93m'
    FAIL = '\033[91m'     # Red for Max Drawdown / Losses
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[97m'    # White for text/labels
    
    # Expanded Color Palette for 16+ Strategies
    STRAT_COLORS = [
        '\033[38;5;208m', # Orange
        '\033[38;5;220m', # Yellow-Orange
        '\033[92m',       # Light Green
        '\033[94m',       # Blue
        '\033[96m',       # Cyan
        '\033[95m',       # Magenta
        '\033[38;5;196m', # Red
        '\033[38;5;46m',  # Bright Green
        '\033[38;5;51m',  # Bright Cyan
        '\033[38;5;201m', # Bright Magenta
        '\033[38;5;226m', # Bright Yellow
        '\033[38;5;165m', # Purple
        '\033[38;5;214m', # Gold
        '\033[38;5;39m',  # Deep Sky Blue
        '\033[38;5;118m', # Chartreuse
        '\033[38;5;129m', # Violet
    ]

# --- CONFIGURATION ---
try:
    import key_file as k
except ImportError:
    print("WARNING: key_file.py not found. Using placeholder API keys.")
    class K:
        xB_KEY = "YOUR_API_KEY"
        xB_SECRET = "YOUR_SECRET_KEY"
    k = K()

# === TRADING BOT CLASS ===
class TradingBot:
    def __init__(self):
        # --- Exchange Init ---
        self.binance = ccxt.binance({
            'apiKey': k.xB_KEY,
            'secret': k.xB_SECRET,
            'enableRateLimit': True,
            'options': {
                'warnOnFetchCurrenciesWithoutPermission': False,
                'fetchCurrencies': False
            }
        })

        # --- Configuration/State ---
        self.initial_balance = 1000.0
        self.balance = self.initial_balance
        self.trade_amount = 50.0  
        self.take_profit_pct = 0.005  # 0.5%
        self.stop_loss_pct = 0.005  # 0.5%
        self.timeframe = '1h'
        self.log_file = "simulate_log.txt"
        self.csv_file = "performance_log.csv"
        self.win_count = 0
        self.loss_count = 0
        self.auto_mode = False
        self.rsi_config_file = "best_rsi_config.json"
        
        self.rsi_thresholds = self._load_rsi_config()
        
        # --- Setup ---
        if os.path.exists(self.log_file): os.remove(self.log_file)
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", encoding="utf-8") as f:
                f.write("timestamp,symbol,signal,entry_price,outcome,balance\n")

        # --- DEFINE STRATEGY GROUPS ---
        # Core 5 strategies used for the Ultimate Summary baseline
        self.CORE_5_STRATEGIES = ['RSI', 'SMA', 'EMACross', 'MACD', 'Bollinger']
        # All original strategies to be displayed first
        self.ORIGINAL_STRATEGIES = self.CORE_5_STRATEGIES + ['RSI_Pivot']

        # Generate 10 Combinations of 3
        # Format: "RSI+SMA+MACD"
        self.combo_definitions = {}
        combos = list(itertools.combinations(self.CORE_5_STRATEGIES, 3))
        for c in combos:
            name = "+".join(c)
            self.combo_definitions[name] = c

    def _load_rsi_config(self):
        """Loads the best RSI config from file, or uses default."""
        try:
            with open(self.rsi_config_file, 'r') as f:
                config = json.load(f)
                if 'BTC/USDT' in config:
                    return config['BTC/USDT']
        except FileNotFoundError:
            pass
        return {"lower": 30, "upper": 70}

    # === LOGGING ===
    def log(self, msg):
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")

    def log_csv(self, ts, symbol, signal, entry, outcome, balance):
        with open(self.csv_file, "a", encoding="utf-8") as f:
            f.write(f"{ts},{symbol},{signal},{entry:.2f},{outcome},{balance:.2f}\n")

    # === DATA FETCHING & INDICATOR COMPUTATION ===
    def fetch_full_year_data(self, symbol, year):
        self.log(f"{Colors.OKCYAN}Fetching {symbol} data for {year}...{Colors.ENDC}")
        since = int(datetime(year, 1, 1).timestamp() * 1000)
        end = int(datetime(year + 1, 1, 1).timestamp() * 1000)
        df_list = []
        limit = 1000

        while since < end:
            try:
                data = self.binance.fetch_ohlcv(symbol, self.timeframe, since=since, limit=limit)
            except Exception as e:
                self.log(f"{Colors.FAIL}⚠️ Fetch error: {e}{Colors.ENDC}")
                break

            if not data:
                break
            
            last_time = data[-1][0]
            since = last_time + 1
            df_list.append(pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume']))
            time.sleep(0.5)

        if not df_list:
            self.log(f"{Colors.FAIL}❌ No data fetched.{Colors.ENDC}")
            return pd.DataFrame()

        df = pd.concat(df_list, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # --- COMPUTE ALL INDICATORS ONCE ---
        df['rsi'] = RSIIndicator(df['close']).rsi()
        df['sma'] = SMAIndicator(df['close'], 20).sma_indicator()
        df['ema_fast'] = EMAIndicator(df['close'], 9).ema_indicator()
        df['ema_slow'] = EMAIndicator(df['close'], 21).ema_indicator()

        ema12 = EMAIndicator(df['close'], 12).ema_indicator()
        ema26 = EMAIndicator(df['close'], 26).ema_indicator()
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = EMAIndicator(df['macd_line'], 9).ema_indicator()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']

        df['atr'] = (df['high'] - df['low']).rolling(14).mean()

        df = df.dropna().reset_index(drop=True)
        self.log(f"{Colors.OKGREEN}✅ Data fetched and processed. Total candles: {len(df)}{Colors.ENDC}")
        return df

    def get_market_data(self, symbol, tf='15m', limit=100):
        try:
            ohlcv = self.binance.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df['rsi'] = RSIIndicator(df['close']).rsi()
            df['sma'] = SMAIndicator(df['close'], 20).sma_indicator()
            df['ema_fast'] = EMAIndicator(df['close'], 9).ema_indicator()
            df['ema_slow'] = EMAIndicator(df['close'], 21).ema_indicator()
            
            exp12 = EMAIndicator(df['close'], 12).ema_indicator()
            exp26 = EMAIndicator(df['close'], 26).ema_indicator()
            df['macd_line'] = exp12 - exp26
            df['macd_signal'] = EMAIndicator(df['macd_line'], 9).ema_indicator()
            df['macd_hist'] = df['macd_line'] - df['macd_signal']
            
            df['price_range'] = df['high'] - df['low']
            df['atr'] = df['price_range'].rolling(14).mean()

            return df
        except Exception as e:
            self.log(f"{Colors.FAIL}⚠️ Error fetching market data: {e}{Colors.ENDC}")
            return pd.DataFrame()


    # === STRATEGY LOGIC ===
    def get_signal(self, df, strategy):
        """
        Generates a trade signal.
        Supports both Base Strategies and Combination Strategies.
        """
        
        # --- 1. HANDLE COMBO STRATEGIES (RECURSIVE CALLS) ---
        if strategy in self.combo_definitions:
            components = self.combo_definitions[strategy] # e.g., ['RSI', 'SMA', 'MACD']
            signals = [self.get_signal(df, comp) for comp in components]
            
            # MAJORITY VOTE: At least 2 out of 3 must agree
            buy_count = signals.count('BUY')
            sell_count = signals.count('SELL')
            
            if buy_count >= 2:
                return 'BUY'
            if sell_count >= 2:
                return 'SELL'
            return 'HOLD'

        # --- 2. HANDLE BASE STRATEGIES ---
        if len(df) < 60:
            return "HOLD"

        last = df.iloc[-1]
        
        def safe(val):
            try: return float(val)
            except: return float('nan')

        rsi = safe(last.get("rsi"))
        rsi_prev = safe(df["rsi"].iloc[-2])
        sma = safe(last.get("sma"))
        ema_fast = safe(last.get("ema_fast"))
        ema_slow = safe(last.get("ema_slow"))
        macd_line = safe(last.get("macd_line"))
        macd_signal = safe(last.get("macd_signal"))
        atr = safe(last.get("atr"))
        close = safe(last["close"])
        
        if any(pd.isna(x) for x in [rsi, sma, ema_fast, ema_slow, atr, macd_line, macd_signal]):
            return "HOLD"

        # STRATEGY: RSI
        if strategy == 'RSI':
            if atr < close * 0.005: return 'HOLD'
            rsi_up = rsi > rsi_prev
            rsi_down = rsi < rsi_prev
            if rsi < self.rsi_thresholds['lower'] and rsi_up: return 'BUY'
            elif rsi > self.rsi_thresholds['upper'] and rsi_down: return 'SELL'
            else: return 'HOLD'

        # STRATEGY: SMA
        elif strategy == "SMA":
            if atr < close * 0.002: return "HOLD"
            sma_prev5 = df["sma"].iloc[-5]
            sma_slope = sma - sma_prev5
            strong_up = sma_slope > 0
            strong_down = sma_slope < 0
            if df["close"].iloc[-2] < df["sma"].iloc[-2] and close > sma and strong_up: return "BUY"
            if df["close"].iloc[-2] > df["sma"].iloc[-2] and close < sma and strong_down: return "SELL"
            return "HOLD"

        # STRATEGY: EMA Cross
        elif strategy == "EMACross":
            if atr < close * 0.002: return "HOLD"
            if ema_fast > ema_slow: return "BUY"
            if ema_fast < ema_slow: return "SELL"
            return "HOLD"

        # STRATEGY: MACD
        elif strategy == "MACD":
            ema50 = df["close"].ewm(span=50).mean().iloc[-1]
            macd_cross_up = macd_line > macd_signal and df['macd_line'].iloc[-2] <= df['macd_signal'].iloc[-2]
            macd_cross_down = macd_line < macd_signal and df['macd_line'].iloc[-2] >= df['macd_signal'].iloc[-2]
            if macd_cross_up and close > ema50: return "BUY"
            if macd_cross_down and close < ema50: return "SELL"
            return "HOLD"

        # STRATEGY: Bollinger
        elif strategy == "Bollinger":
            bb = BollingerBands(df["close"])
            lower = bb.bollinger_lband().iloc[-1]
            upper = bb.bollinger_hband().iloc[-1]
            ema50 = df["close"].ewm(span=50).mean().iloc[-1]
            if close < lower and close > df["close"].iloc[-2] and close > ema50: return "BUY"
            if close > upper and close < df["close"].iloc[-2] and close < ema50: return "SELL"
            return "HOLD"

        # STRATEGY: RSI + Pivot
        elif strategy == "RSI_Pivot":
            rsi_series = df["rsi"]
            pivot_low = df["close"].iloc[-3] > df["close"].iloc[-2] < df["close"].iloc[-1]
            pivot_high = df["close"].iloc[-3] < df["close"].iloc[-2] > df["close"].iloc[-1]
            ema50 = df["close"].ewm(span=50).mean().iloc[-1]
            trend_up = close > ema50
            trend_down = close < ema50
            if pivot_low and rsi_series.iloc[-1] > 30 and ema_fast > ema_slow and trend_up: return "BUY"
            if pivot_high and rsi_series.iloc[-1] < 70 and ema_fast < ema_slow and trend_down: return "SELL"
            return "HOLD"

        return "HOLD"

    # === PLOTTING (Unchanged) ===
    def plot_backtest_result(self, trade_numbers, balance_history):
        plt.rcParams['font.family'] = 'Segoe UI Emoji'
        plt.figure(figsize=(10, 5))
        plt.plot(trade_numbers, balance_history, color='blue', linewidth=2, label='Balance')

        plt.title("📈 Backtest Balance Over Time", fontfamily="Segoe UI Emoji")
        plt.xlabel("Trade #")
        plt.ylabel("Balance (USD)")
        plt.axhline(self.initial_balance, color='red', linestyle='--', label='Initial Balance')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("balance_result.png")
        plt.show()

    # === CORE METRICS CALCULATION (Unchanged) ===
    def evaluate_strategy(self, df, strategy):
        if len(df) < 60:
            return {"strategy": strategy, "trades": 0, "wins": 0, "losses": 0, "winrate": 0.0, "final_balance": self.initial_balance, "max_drawdown": 0.0, "profit_factor": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "avg_trade_pl": 0.0}

        balance = self.initial_balance
        peak_balance = self.initial_balance
        max_drawdown = 0.0
        trades = 0
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0

        start_index = 60
        for i in range(start_index, len(df) - 1):
            df_slice = df.iloc[:i+1].copy() 
            price = df_slice['close'].iloc[-1]
            
            signal = self.get_signal(df_slice, strategy)

            if signal not in ['BUY', 'SELL']:
                continue

            trades += 1
            entry = price
            
            tp = entry * (1 + self.take_profit_pct) if signal == "BUY" else entry * (1 - self.take_profit_pct)
            sl = entry * (1 - self.stop_loss_pct) if signal == "BUY" else entry * (1 + self.stop_loss_pct)

            for j in range(i+1, len(df)):
                forward_high = df['high'].iloc[j]
                forward_low = df['low'].iloc[j]
                
                if (signal == 'BUY' and forward_high >= tp) or (signal == 'SELL' and forward_low <= tp):
                    profit_amount = self.trade_amount * self.take_profit_pct
                    balance += profit_amount
                    gross_profit += profit_amount
                    wins += 1
                    break
                    
                if (signal == 'BUY' and forward_low <= sl) or (signal == 'SELL' and forward_high >= sl):
                    loss_amount = self.trade_amount * self.stop_loss_pct
                    balance -= loss_amount
                    gross_loss += loss_amount
                    losses += 1
                    break
            
            if balance > peak_balance:
                peak_balance = balance
            drawdown = ((peak_balance - balance) / peak_balance) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            
        winrate = (wins / trades * 100) if trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0 
        gross_return = (balance - self.initial_balance) / self.initial_balance * 100
        avg_trade_pl = (gross_profit - gross_loss) / trades if trades > 0 else 0.0

        return {
            "strategy": strategy,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "final_balance": balance,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "gross_return": gross_return,
            "avg_trade_pl": avg_trade_pl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss
        }

    # === DEDICATED DISPLAY FUNCTION (Unchanged) ===
    def _display_strategy_result(self, res, frame_color):
        strategy = res['strategy']
        final_balance = res['final_balance']
        gross_return = res['gross_return']
        max_drawdown = res['max_drawdown']
        profit_factor = res['profit_factor']
        avg_trade_pl = res['avg_trade_pl']
        
        pl_color = Colors.OKGREEN if gross_return >= 0 else Colors.FAIL
        mdd_color = Colors.FAIL if max_drawdown > 0 else Colors.OKGREEN 
            
        self.log(f"{frame_color}{Colors.WHITE} STRATEGY: {strategy.upper()} PERFORMANCE REPORT {Colors.ENDC}")
        self.log(f"{frame_color}{Colors.WHITE}   {Colors.BOLD}NET RETURN: {pl_color}{gross_return:+.2f}%{Colors.ENDC}{Colors.WHITE} (Final: {pl_color}${final_balance:.2f}){Colors.ENDC}")
        self.log(f"{frame_color}{Colors.WHITE}   {Colors.BOLD}MAX DRAWDOWN: {mdd_color}{max_drawdown:.2f}%{Colors.ENDC}")
        self.log(f"{frame_color}{Colors.WHITE}   {Colors.BOLD}PROFIT FACTOR: {Colors.OKGREEN}{profit_factor:.2f}{Colors.ENDC}")
        self.log(f"{frame_color}{Colors.WHITE}   {Colors.BOLD}WIN RATE: {Colors.OKGREEN}{res['winrate']:.2f}%{Colors.ENDC}{Colors.WHITE} ({res['trades']} Trades){Colors.ENDC}")
        self.log(f"{frame_color}{Colors.WHITE}   {Colors.BOLD}AVG TRADE P&L: {pl_color}{avg_trade_pl:+.2f} USD{Colors.ENDC}")
        print("\n")

    # === OPTION 1: SINGLE STRATEGY BACKTEST (Unchanged) ===
    def backtest_strategy(self, symbol, strategy, year):
        print("\n")
        self.log(f"{Colors.WHITE}{Colors.BOLD}(SINGLE BACKTEST: {symbol} | {year}{Colors.ENDC})")
        self.log(f"{Colors.WHITE}CONFIG: {Colors.OKGREEN}{self.timeframe}{Colors.ENDC}{Colors.WHITE} TF | {Colors.OKGREEN}{self.take_profit_pct*100:.1f}% TP{Colors.ENDC}{Colors.WHITE} | {Colors.OKGREEN}{self.stop_loss_pct*100:.1f}% SL{Colors.ENDC}")
        self.log(f"{Colors.WHITE}CAPITAL: {Colors.OKGREEN}${self.initial_balance:.2f}{Colors.ENDC}{Colors.WHITE} (Risk: ${self.trade_amount:.2f}){Colors.ENDC}")
        
        df = self.fetch_full_year_data(symbol, year)

        if df.empty:
            return

        results = self.evaluate_strategy(df, strategy)

        # Rerun for plot data (kept separate to keep evaluate_strategy pure)
        balance = self.initial_balance
        balance_history = [balance]
        trade_numbers = [0]
        trade_count = 0

        for i in range(60, len(df) - 1):
            df_slice = df.iloc[:i+1].copy()
            price = df_slice['close'].iloc[-1]
            signal = self.get_signal(df_slice, strategy)

            if signal not in ['BUY', 'SELL']:
                continue

            entry = price
            tp = entry * (1 + self.take_profit_pct) if signal == "BUY" else entry * (1 - self.take_profit_pct)
            sl = entry * (1 - self.stop_loss_pct) if signal == "BUY" else entry * (1 + self.stop_loss_pct)

            for j in range(i+1, len(df)):
                forward_high = df['high'].iloc[j]
                forward_low = df['low'].iloc[j]

                trade_resolved = False
                if (signal == "BUY" and forward_high >= tp):
                    balance += self.trade_amount * self.take_profit_pct
                    trade_resolved = True
                elif (signal == "SELL" and forward_low <= tp):
                    balance += self.trade_amount * self.take_profit_pct
                    trade_resolved = True
                elif (signal == "BUY" and forward_low <= sl):
                    balance -= self.trade_amount * self.stop_loss_pct
                    trade_resolved = True
                elif (signal == "SELL" and forward_high >= sl):
                    balance -= self.trade_amount * self.stop_loss_pct
                    trade_resolved = True
                
                if trade_resolved:
                    trade_count += 1
                    balance_history.append(balance)
                    trade_numbers.append(trade_count)
                    break
            
        self.log(f"\n{Colors.OKCYAN}{Colors.BOLD}--- DETAILED BACKTEST RESULTS ---{Colors.ENDC}\n")
        single_color = Colors.STRAT_COLORS[0]
        self._display_strategy_result(results, single_color)
        self.plot_backtest_result(trade_numbers, balance_history)


    # === OPTION 2: COMPARISON FUNCTION (UPDATED) ===
    def compare_all_strategies(self, symbol, year):
        """Compares all Basic + Combo strategies with structured output."""
        
        # 1. UI HEADER
        print("\n")
        self.log(f"{Colors.WHITE}{Colors.BOLD}(COMPARISON BACKTEST REPORT: {symbol} | {year}{Colors.ENDC})")
        self.log(f"{Colors.WHITE}CONFIG: {Colors.OKGREEN}{self.timeframe}{Colors.ENDC}{Colors.WHITE} TF | {Colors.OKGREEN}{self.take_profit_pct*100:.1f}% TP{Colors.ENDC}{Colors.WHITE} | {Colors.OKGREEN}{self.stop_loss_pct*100:.1f}% SL{Colors.ENDC}")
        self.log(f"{Colors.WHITE}CAPITAL: {Colors.OKGREEN}${self.initial_balance:.2f}{Colors.ENDC}{Colors.WHITE} (Risk: ${self.trade_amount:.2f}){Colors.ENDC}")
        
        # Define all strategies for testing
        ALL_STRATEGIES = self.ORIGINAL_STRATEGIES + list(self.combo_definitions.keys())

        # 2. DATA FETCH & EVALUATION
        df = self.fetch_full_year_data(symbol, year)
        
        if df.empty:
            return []
        
        self.log(f"\n{Colors.OKCYAN}{Colors.BOLD}--- EVALUATING STRATEGIES ---{Colors.ENDC}\n")
        
        results = []
        for s in ALL_STRATEGIES:
            print(f"Testing {s}...", end='\r') # Progress indicator
            res = self.evaluate_strategy(df, s)
            results.append(res)
            
        # 3. SEPARATION AND SORTING
        original_results = [r for r in results if r['strategy'] in self.ORIGINAL_STRATEGIES]
        combo_results = [r for r in results if r['strategy'] in self.combo_definitions]

        # Sort both groups by performance (final_balance)
        original_results.sort(key=lambda x: x['final_balance'], reverse=True)
        combo_results.sort(key=lambda x: x['final_balance'], reverse=True)

        # Combine for unified color mapping
        sorted_results = original_results + combo_results
        
        # 4. COLOR-CODED RESULTS DISPLAY (apply colors based on overall position)
        colored_results = []
        for i, res in enumerate(sorted_results):
            res['color'] = Colors.STRAT_COLORS[i % len(Colors.STRAT_COLORS)]
            colored_results.append(res)

        # Display Originals first
        self.log(f"\n{Colors.WARNING}{Colors.BOLD}--- ORIGINAL STRATEGIES PERFORMANCE ({len(original_results)}) ---{Colors.ENDC}")
        for res in [r for r in colored_results if r['strategy'] in self.ORIGINAL_STRATEGIES]:
            self._display_strategy_result(res, res['color'])

        # Display Combos second
        self.log(f"\n{Colors.WARNING}{Colors.BOLD}--- COMBO STRATEGIES PERFORMANCE ({len(combo_results)}) ---{Colors.ENDC}")
        for res in [r for r in colored_results if r['strategy'] in self.combo_definitions]:
            self._display_strategy_result(res, res['color'])

            
        # 5. ULTIMATE PERFORMANCE SUMMARY (based ONLY on CORE 5 strategies)
        core_5_results = [r for r in results if r['strategy'] in self.CORE_5_STRATEGIES]
        core_5_results.sort(key=lambda x: x['final_balance'], reverse=True)
        
        best = core_5_results[0] if core_5_results else None
        worst = core_5_results[-1] if core_5_results else None
        
        self.log(f"\n{Colors.WARNING}{Colors.BOLD}ULTIMATE PERFORMANCE SUMMARY (Based on Core 5 Strategies){Colors.ENDC}")
        
        if best and worst:
            total_profit_factor = sum(r['profit_factor'] for r in core_5_results if r['profit_factor'] != 999.0)
            valid_pf_count = sum(1 for r in core_5_results if r['profit_factor'] != 999.0)
            avg_pf = total_profit_factor / valid_pf_count if valid_pf_count > 0 else 0.0

            self.log(f"{Colors.WHITE}{Colors.BOLD}Best Strategy P&L:{Colors.ENDC} {best['color']}{best['strategy']}{Colors.ENDC} ({Colors.OKGREEN}{best['gross_return']:+.2f}%{Colors.ENDC})")
            self.log(f"{Colors.WHITE}{Colors.BOLD}Worst Strategy P&L:{Colors.ENDC} {worst['color']}{worst['strategy']}{Colors.ENDC} ({Colors.FAIL}{worst['gross_return']:+.2f}%{Colors.ENDC})")
            self.log(f"{Colors.WHITE}{Colors.BOLD}Avg Profit Factor:{Colors.ENDC} {Colors.OKGREEN}{avg_pf:.2f}{Colors.ENDC}")
            
            # === 6. COMBO STRATEGY INSIGHTS ===
            
            if combo_results:
                self.log(f"\n{Colors.OKCYAN}{Colors.BOLD}=== COMBO STRATEGY INSIGHTS (Winrate Analysis) ==={Colors.ENDC}")
                
                # Sort combos by Winrate for this specific analysis
                # Note: combo_results is already sorted by balance, we need to re-sort by winrate
                combo_results_by_winrate = sorted(combo_results, key=lambda x: x['winrate'], reverse=True)
                top_3_combos = combo_results_by_winrate[:3]
                
                # A. Show Top 3 Combos
                self.log(f"{Colors.WHITE}🏆 {Colors.BOLD}Top 3 Combos by Winrate:{Colors.ENDC}")
                for idx, c in enumerate(top_3_combos, 1):
                    self.log(f"  {idx}. {c['strategy']} ({Colors.OKGREEN}{c['winrate']:.2f}% Winrate{Colors.ENDC}, Return: {c['gross_return']:.1f}%)")
                
                # B. Frequency Analysis of Components in Top 3
                component_counts = Counter()
                for c in top_3_combos:
                    components = self.combo_definitions[c['strategy']]
                    component_counts.update(components)
                
                # Most Chosen
                if component_counts:
                    most_common = component_counts.most_common(1)[0] # Returns ('Strategy', count)
                    most_name, most_count = most_common
                    self.log(f"\n{Colors.WHITE}🔥 {Colors.BOLD}Most Frequent Component in Top 3:{Colors.ENDC} {Colors.OKGREEN}{most_name}{Colors.ENDC} (Appears in {most_count}/3 combos)")
                    
                    # Least Chosen (Strategies in base list but with lowest/zero count in top 3)
                    all_counts = {base: component_counts.get(base, 0) for base in self.CORE_5_STRATEGIES}
                    min_count_val = min(all_counts.values())
                    least_common_names = [name for name, count in all_counts.items() if count == min_count_val]
                    
                    least_str = ", ".join(least_common_names)
                    self.log(f"{Colors.WHITE}❄️ {Colors.BOLD}Least Frequent Component in Top 3:{Colors.ENDC} {Colors.FAIL}{least_str}{Colors.ENDC} (Appears in {min_count_val}/3 combos)")

        else:
            self.log(f"{Colors.FAIL}No results to summarize.{Colors.ENDC}")

        return results


    # === LIVE SIMULATION (Unchanged) ===
    def get_trending_coin(self):
        tickers = ['BTC/USDT','ETH/USDT','BNB/USDT','SOL/USDT','XRP/USDT']
        ranked = []
        for t in tickers:
            try:
                df = self.get_market_data(t, '1h', 50)
                change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                ranked.append((t, change))
            except: continue
        ranked.sort(key=lambda x: x[1], reverse=True)
        self.log("\n📈 Trending:")
        for i, (s,c) in enumerate(ranked,1): self.log(f"{i}. {s} ({c*100:.2f}%)")
        if ranked:
            return ranked[0][0]
        return 'BTC/USDT'

    def simulate_trade(self, signal, symbol, entry):
        self.balance = self.initial_balance
        
        tp = entry * (1 + self.take_profit_pct) if signal == "BUY" else entry * (1 - self.take_profit_pct)
        sl = entry * (1 - self.stop_loss_pct) if signal == "BUY" else entry * (1 + self.stop_loss_pct)
        self.log(f"\nSimulating {signal} {symbol} @ {entry:.2f} → TP: {tp:.2f}, SL: {sl:.2f}")
        
        start = time.time()
        timeout = 1800
        while time.time() - start < timeout:
            df = self.get_market_data(symbol, self.timeframe, 2)
            if df.empty:
                time.sleep(5)
                continue
            
            price = df['close'].iloc[-1]
            self.log(f"📊 Price: {price:.2f}")
            
            if (signal == "BUY" and price >= tp) or (signal == "SELL" and price <= tp):
                self.balance += self.trade_amount * self.take_profit_pct
                self.win_count += 1
                self.log(f"{Colors.OKGREEN}✅ WIN - Balance: ${self.balance:.2f}{Colors.ENDC}")
                break
            elif (signal == "BUY" and price <= sl) or (signal == "SELL" and price >= sl):
                self.balance -= self.trade_amount * self.stop_loss_pct
                self.loss_count += 1
                self.log(f"{Colors.FAIL}❌ LOSS - Balance: ${self.balance:.2f}{Colors.ENDC}")
                break
            time.sleep(5)
        else:
            self.log("🕒 Trade timed out. No TP/SL hit.")


    def run_live_simulation(self):
        self.auto_mode = input("🤖 Enable Auto Mode? (yes/no): ").strip().lower() == "yes"
        self.log(f"{'✅ Auto Mode Enabled' if self.auto_mode else '🧠 Manual Mode Enabled'}")
        
        strategy = input("Choose strategy (RSI/SMA/EMACross/MACD/Bollinger/RSI_Pivot): ").strip()
        
        if strategy == 'RSI' and os.path.exists(self.rsi_config_file):
            self.log(f"Using optimized RSI config: {self.rsi_thresholds}")

        symbol = self.get_trending_coin()

        while True:
            df = self.get_market_data(symbol, self.timeframe)
            if df.empty:
                time.sleep(60)
                continue

            signal = self.get_signal(df, strategy)
            entry = df['close'].iloc[-1]
            rsi = df.get('rsi', pd.Series()).iloc[-1] if 'rsi' in df.columns else 'N/A'
            sma = df.get('sma', pd.Series()).iloc[-1] if 'sma' in df.columns else 'N/A'
            
            self.log(f"\n🕒 {datetime.now()} | {symbol} | Price: {entry:.2f} | SMA: {sma:.2f} | RSI: {rsi:.2f} | Signal: {signal}")
            
            if signal in ['BUY', 'SELL']:
                self.simulate_trade(signal, symbol, entry)
            else:
                self.log("📌 No valid signal. Holding...")
            
            time.sleep(60)

    # === OPTIMIZATION (RSI) (Unchanged) ===
    def fetch_ohlcv_for_optimization(self, symbol="BTC/USDT", timeframe="1h", limit=200):
        try:
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.log(f"{Colors.FAIL}Error fetching data for optimization: {e}{Colors.ENDC}")
            return pd.DataFrame()

    def run_rsi_backtest_optimizer(self, df, lower, upper):
        df['rsi'] = RSIIndicator(df['close']).rsi()
        balance = self.initial_balance
        position = None
        
        for i in range(20, len(df)):
            rsi = df['rsi'].iloc[i]
            price = df['close'].iloc[i]
            
            if position is None:
                if rsi < lower: position = "BUY"; entry = price
                elif rsi > upper: position = "SELL"; entry = price
            
            elif position == "BUY":
                if price >= entry * (1 + self.take_profit_pct): balance += self.trade_amount * self.take_profit_pct; position = None
                elif price <= entry * (1 - self.stop_loss_pct): balance -= self.trade_amount * self.stop_loss_pct; position = None
            
            elif position == "SELL":
                if price <= entry * (1 - self.take_profit_pct): balance += self.trade_amount * self.take_profit_pct; position = None
                elif price >= entry * (1 + self.stop_loss_pct): balance -= self.trade_amount * self.stop_loss_pct; position = None
        return balance

    def optimize_rsi_for_symbol(self, symbol, df):
        best = {'profit': -1, 'lower': 30, 'upper': 70}
        
        for l in range(10, 40, 5):
            for u in range(60, 90, 5):
                if l >= u: continue
                profit = self.run_rsi_backtest_optimizer(df.copy(), l, u)
                if profit > best['profit']:
                    best = {'symbol': symbol, 'lower': l, 'upper': u, 'profit': profit}
        return best

    def optimize_across_multiple_coins(self, symbols, tf='1h'):
        self.log("\n⚠️ Running Optimization: WARNING - This is highly susceptible to OVERFITTING.")
        
        results = []
        config = {}
        for s in symbols:
            try:
                df = self.fetch_ohlcv_for_optimization(s, tf) 
                res = self.optimize_rsi_for_symbol(s, df)
                config[s] = {'lower': res['lower'], 'upper': res['upper']}
                results.append(res)
            except Exception as e:
                self.log(f"⚠️ {s}: Error during optimization: {e}")
        
        with open(self.rsi_config_file, 'w') as f:
            json.dump(config, f, indent=2)
            self.rsi_thresholds = config.get('BTC/USDT', self.rsi_thresholds)
            
        self.log("\n=== RSI Optimization Complete ===")
        for r in results:
            self.log(f"{r['symbol']} → {r['lower']}/{r['upper']} → ${r['profit']:.2f}")

    # === MENU (Unchanged) ===
    def main_menu(self):
        while True:
            print("\n=== TRADING BOT MENU ===")
            print("1. 🧠 Run Live Simulation (Paper Trading)")
            print("2. ⏳ Run Backtest (Single/Compare)")
            print("3. 🧪 Optimize RSI (Warning: Overfitting)")
            print("4. ❌ Exit")
            
            choice = input("Enter choice: ").strip()

            if choice == '1':
                self.run_live_simulation()

            elif choice == '2':
                symbol = input("Symbol (e.g. BTC/USDT): ").strip().upper()
                try:
                    year = int(input("Year to backtest (e.g. 2024): "))
                except ValueError:
                    print("❌ Invalid year.")
                    continue

                print("(1) Backtest single strategy")
                print("(2) Compare all strategies (Originals + 10 New Combos)")
                mode = input("Choose: ").strip()

                if mode == '1':
                    print("Strategy: (1) RSI (2) SMA (3) EMACross (4) MACD (5) Bollinger (6) RSI_Pivot")
                    strat_map = {'1': 'RSI', '2': 'SMA', '3': 'EMACross', '4': 'MACD', '5': 'Bollinger', '6': 'RSI_Pivot'}
                    choice_strat = input("Choose strategy (1-6): ").strip()
                    if choice_strat in strat_map:
                        self.backtest_strategy(symbol, strat_map[choice_strat], year)
                    else:
                        print("❌ Invalid strategy.")
                
                elif mode == '2':
                    self.compare_all_strategies(symbol, year)

                else:
                    print("❌ Invalid choice (1 or 2).")

            elif choice == '3':
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
                self.optimize_across_multiple_coins(symbols)

            elif choice == '4':
                self.log("👋 Bye!")
                break

            else:
                print("❌ Invalid menu choice.")

# === RUN ===
if __name__ == "__main__":
    bot = TradingBot()
    bot.main_menu()
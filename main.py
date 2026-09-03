# ============================================================
# ربات حرفه‌ای تحلیل بورس تهران - نسخه کامل با بک‌تست و پیش‌بینی
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler
import algotik_tse as tse
import ta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# تنظیم لاگینگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv('BOT_TOKEN')
if not TOKEN:
    raise ValueError("❌ توکن ربات یافت نشد!")

# ===================== کلاس تحلیلگر پیشرفته ==========================

class AdvancedAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.df = None

    def fetch_data(self, days=300):
        try:
            self.df = tse.get_history(symbol=self.symbol, adjust=True, include_jdate=True)
            if self.df is None or self.df.empty:
                return False
            if len(self.df) > days:
                self.df = self.df.tail(days)
            self.df = self.df.sort_index()
            return True
        except Exception as e:
            logger.error(f"خطا در دریافت داده: {e}")
            return False

    def calculate_all_indicators(self):
        if self.df is None or self.df.empty:
            return False

        df = self.df.copy()
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values

        # MACD
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # Stochastic
        df['Stoch_K'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
        df['Stoch_D'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch_signal()

        # OBV (On-Balance Volume)
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # میانگین متحرک
        df['SMA_20'] = pd.Series(close).rolling(20).mean()
        df['SMA_50'] = pd.Series(close).rolling(50).mean()
        df['SMA_200'] = pd.Series(close).rolling(200).mean()

        # حجم
        df['Volume_MA_20'] = pd.Series(volume).rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']

        # ورود/خروج پول حقیقی
        try:
            client = tse.get_client_type(symbol=self.symbol, include_jdate=True)
            if client is not None and not client.empty:
                df['Buy_Real'] = client['Buy_Real'].values if 'Buy_Real' in client else 0
                df['Sell_Real'] = client['Sell_Real'].values if 'Sell_Real' in client else 0
                df['Net_Real'] = df['Buy_Real'] - df['Sell_Real']
                df['Net_Real_MA_5'] = df['Net_Real'].rolling(5).mean()
        except:
            pass

        self.df = df
        return True

    def get_full_analysis(self):
        if self.df is None or self.df.empty:
            return "❌ داده‌ای برای تحلیل وجود ندارد."

        last = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) > 1 else last

        # قیمت‌ها
        current_price = last['Close']
        price_change = ((current_price - prev['Close']) / prev['Close']) * 100

        # MACD
        macd = last['MACD']
        macd_signal = last['MACD_Signal']
        if pd.isna(macd) or pd.isna(macd_signal):
            macd_status = "⚠️ داده کافی نیست"
            macd_signal_text = ""
        elif macd > macd_signal:
            macd_status = "✅ صعودی"
            macd_signal_text = "🔺 سیگنال خرید"
        else:
            macd_status = "❌ نزولی"
            macd_signal_text = "🔻 سیگنال فروش"

        # RSI
        rsi = last['RSI']
        if pd.isna(rsi):
            rsi_status = "⚠️ داده کافی نیست"
        elif rsi > 70:
            rsi_status = f"⚠️ اشباع خرید ({rsi:.1f})"
        elif rsi < 30:
            rsi_status = f"✅ اشباع فروش ({rsi:.1f})"
        else:
            rsi_status = f"⚖️ خنثی ({rsi:.1f})"

        # Stochastic
        stoch_k = last.get('Stoch_K', 50)
        stoch_d = last.get('Stoch_D', 50)
        if not pd.isna(stoch_k) and not pd.isna(stoch_d):
            if stoch_k < 20 and stoch_d < 20:
                stoch_status = "✅ اشباع فروش (زمان خرید)"
            elif stoch_k > 80 and stoch_d > 80:
                stoch_status = "⚠️ اشباع خرید (زمان فروش)"
            else:
                stoch_status = f"⚖️ خنثی (K={stoch_k:.1f}, D={stoch_d:.1f})"
        else:
            stoch_status = "⚠️ داده کافی نیست"

        # حجم
        vol_ratio = last['Volume_Ratio']
        if pd.isna(vol_ratio):
            vol_status = "⚠️ داده کافی نیست"
        elif vol_ratio > 2:
            vol_status = f"🔥 بسیار بالا ({vol_ratio:.1f}x)"
        elif vol_ratio > 1.5:
            vol_status = f"📈 بالاتر از میانگین ({vol_ratio:.1f}x)"
        elif vol_ratio > 0.8:
            vol_status = f"📊 عادی ({vol_ratio:.1f}x)"
        else:
            vol_status = f"📉 پایین‌تر از میانگین ({vol_ratio:.1f}x)"

        # میانگین متحرک
        sma_20 = last['SMA_20']
        sma_50 = last['SMA_50']
        if not pd.isna(sma_20) and not pd.isna(sma_50):
            if current_price > sma_20 > sma_50:
                ma_status = "🟢 روند صعودی قوی"
            elif current_price < sma_20 < sma_50:
                ma_status = "🔴 روند نزولی قوی"
            else:
                ma_status = "🟡 روند خنثی"
        else:
            ma_status = "⚠️ داده کافی نیست"

        # ورود/خروج پول
        net_real = last.get('Net_Real', 0)
        net_real_ma = last.get('Net_Real_MA_5', 0)
        if net_real != 0:
            if net_real > 0 and net_real > net_real_ma:
                money_status = f"💰 ورود پول قوی ({net_real:,.0f})"
            elif net_real > 0:
                money_status = f"💰 ورود پول ({net_real:,.0f})"
            elif net_real < 0 and net_real < net_real_ma:
                money_status = f"💸 خروج پول شدید ({abs(net_real):,.0f})"
            else:
                money_status = f"💸 خروج پول ({abs(net_real):,.0f})"
        else:
            money_status = "⚠️ در دسترس نیست"

        # امتیازدهی نهایی
        score = 0
        signals = []

        if not pd.isna(macd) and not pd.isna(macd_signal):
            if macd > macd_signal:
                score += 1
                signals.append("MACD خرید")
            else:
                score -= 1
                signals.append("MACD فروش")

        if not pd.isna(rsi):
            if rsi < 30:
                score += 1
                signals.append("RSI اشباع فروش")
            elif rsi > 70:
                score -= 1
                signals.append("RSI اشباع خرید")

        if not pd.isna(vol_ratio) and vol_ratio > 1.5:
            score += 0.5
            signals.append("حجم بالا")

        if net_real > 0:
            score += 1
            signals.append("ورود پول")
        elif net_real < 0:
            score -= 1
            signals.append("خروج پول")

        if score >= 2:
            overall = "🟢 **روند صعودی قوی** - زمان خرید مناسب"
        elif score >= 0.5:
            overall = "🟡 **روند خنثی تا صعودی** - با احتیاط خرید"
        elif score >= -1:
            overall = "🟠 **روند خنثی تا نزولی** - ریسک بالا"
        else:
            overall = "🔴 **روند نزولی قوی** - از خرید خودداری کنید"

        # ساخت خروجی
        text = f"""
📊 **تحلیل حرفه‌ای {self.symbol}**
📅 تاریخ: {last.get('JDate', 'نامشخص')}

━━━━━━━━━━━━━━━━━━━━━━━━━━
💹 **قیمت‌ها**
   قیمت پایانی: {current_price:,.0f} تومان
   تغییر روزانه: {price_change:+.2f}%
   بیشترین: {last['High']:,.0f} | کمترین: {last['Low']:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **شاخص‌های تکنیکال**
   MACD: {macd_status} {macd_signal_text}
   RSI: {rsi_status}
   Stochastic: {stoch_status}
   میانگین متحرک: {ma_status}

━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **حجم و پول**
   حجم: {vol_status}
   ورود/خروج پول: {money_status}

━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **جمع‌بندی نهایی**
   امتیاز: {score:.1f}/4
   {overall}
   سیگنال‌ها: {', '.join(signals) if signals else 'بدون سیگنال خاص'}
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return text

    def backtest(self, days=100, initial_capital=100000000):
        if self.df is None or len(self.df) < days:
            return None

        df = self.df.tail(days).copy()
        signals = pd.Series(0, index=df.index)

        # استراتژی ترکیبی: RSI + MACD
        buy_condition = (df['RSI'] < 35) & (df['MACD'] > df['MACD_Signal'])
        sell_condition = (df['RSI'] > 65) & (df['MACD'] < df['MACD_Signal'])
        signals[buy_condition] = 1
        signals[sell_condition] = -1

        position = 0
        capital = initial_capital
        shares = 0
        trades = []

        for i in range(1, len(signals)):
            if signals.iloc[i] == 1 and position == 0:
                price = df['Close'].iloc[i]
                shares = capital // price
                capital -= shares * price
                position = 1
                trades.append(('خرید', df.index[i], price))
            elif signals.iloc[i] == -1 and position == 1:
                price = df['Close'].iloc[i]
                capital += shares * price
                shares = 0
                position = 0
                trades.append(('فروش', df.index[i], price))

        if position == 1:
            price = df['Close'].iloc[-1]
            capital += shares * price
            trades.append(('فروش (نهایی)', df.index[-1], price))

        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100
        num_trades = len(trades) // 2

        win_count = 0
        if num_trades > 0:
            for j in range(0, len(trades)-1, 2):
                if trades[j+1][2] > trades[j][2]:
                    win_count += 1
            win_rate = (win_count / num_trades) * 100
        else:
            win_rate = 0

        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

        # نمایش جزئیات معاملات
        trades_text = ""
        for i, trade in enumerate(trades, 1):
            trades_text += f"{i}. {trade[0]} در تاریخ {trade[1]} به قیمت {trade[2]:,.0f}\n"

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'buy_hold_return': buy_hold_return,
            'trades': trades_text
        }

    def predict_price(self, days_ahead=5):
        if self.df is None or len(self.df) < 30:
            return None

        df = self.df.copy()
        for lag in range(1, 6):
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        df['RSI_lag'] = df['RSI'].shift(1)
        df['MACD_lag'] = df['MACD'].shift(1)
        df['Volume_lag'] = df['Volume'].shift(1)
        df = df.dropna()

        if len(df) < 20:
            return None

        X = df[['Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_4', 'Close_lag_5',
                'RSI_lag', 'MACD_lag', 'Volume_lag']].values
        y = df['Close'].shift(-1).dropna().values

        X = X[:-1]
        y = y
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)

        last_row = df.iloc[-1]
        predictions = []
        current_features = [
            last_row['Close_lag_1'], last_row['Close_lag_2'], last_row['Close_lag_3'],
            last_row['Close_lag_4'], last_row['Close_lag_5'],
            last_row['RSI_lag'], last_row['MACD_lag'], last_row['Volume_lag']
        ]

        for _ in range(days_ahead):
            pred_price = model.predict(scaler.transform([current_features]))[0]
            predictions.append(pred_price)
            current_features = [pred_price, current_features[0], current_features[1],
                               current_features[2], current_features[3],
                               last_row['RSI'], last_row['MACD'], last_row['Volume']]

        trend = []
        for i in range(1, len(predictions)):
            trend.append('صعودی 📈' if predictions[i] > predictions[i-1] else 'نزولی 📉')

        return {
            'predictions': predictions,
            'trend': trend,
            'current_price': self.df['Close'].iloc[-1]
        }

    def get_signals(self):
        """زمان‌های پیشنهادی خرید/فروش بر اساس ترکیب شاخص‌ها"""
        if self.df is None or len(self.df) < 50:
            return None

        df = self.df.tail(50).copy()
        signals = []

        for i in range(1, len(df)):
            rsi = df['RSI'].iloc[i]
            macd = df['MACD'].iloc[i]
            macd_signal = df['MACD_Signal'].iloc[i]
            vol_ratio = df['Volume_Ratio'].iloc[i]
            net_real = df.get('Net_Real', pd.Series([0]*len(df))).iloc[i]

            if (rsi < 35 and macd > macd_signal and vol_ratio > 1.2 and net_real > 0):
                signals.append(('🟢 خرید', df.index[i], df['Close'].iloc[i]))
            elif (rsi > 65 and macd < macd_signal and vol_ratio > 1.2 and net_real < 0):
                signals.append(('🔴 فروش', df.index[i], df['Close'].iloc[i]))

        return signals[-10:] if len(signals) > 10 else signals

# ===================== منوی اصلی با دکمه ==========================

def main_menu():
    keyboard = [
        [InlineKeyboardButton("📊 تحلیل لحظه‌ای", callback_data='analyze')],
        [InlineKeyboardButton("📈 بک‌تست (۱۰۰ روزه)", callback_data='backtest')],
        [InlineKeyboardButton("🔮 پیش‌بینی قیمت", callback_data='predict')],
        [InlineKeyboardButton("⏰ زمان معامله", callback_data='signals')],
        [InlineKeyboardButton("📋 راهنما", callback_data='help')]
    ]
    return InlineKeyboardMarkup(keyboard)

# ===================== دستورات ربات ==========================

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🤖 **ربات حرفه‌ای تحلیل بورس تهران**\n\n"
        "لطفاً یکی از گزینه‌های زیر را انتخاب کنید:",
        reply_markup=main_menu()
    )

def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    if query.data == 'analyze':
        query.edit_message_text("🔍 لطفاً نماد را وارد کنید:\nمثال: `فولاد`")
        context.user_data['action'] = 'analyze'

    elif query.data == 'backtest':
        query.edit_message_text("📈 لطفاً نماد و تعداد روز را وارد کنید:\nمثال: `فولاد 100`")
        context.user_data['action'] = 'backtest'

    elif query.data == 'predict':
        query.edit_message_text("🔮 لطفاً نماد را وارد کنید:\nمثال: `فولاد`")
        context.user_data['action'] = 'predict'

    elif query.data == 'signals':
        query.edit_message_text("⏰ لطفاً نماد را وارد کنید:\nمثال: `فولاد`")
        context.user_data['action'] = 'signals'

    elif query.data == 'help':
        help_text = """
📖 **راهنمای کامل**

🔹 `/analyze <نماد>` - تحلیل کامل با RSI، MACD، Stochastic، حجم، پول و میانگین متحرک
🔹 `/backtest <نماد> <روز>` - بک‌تست استراتژی روی تعداد روز مشخص
🔹 `/predict <نماد>` - پیش‌بینی قیمت برای ۵ روز آینده
🔹 `/signals <نماد>` - نمایش زمان‌های پیشنهادی خرید/فروش
🔹 `/start` - منوی اصلی

📌 **نمادهای معتبر**:
فولاد، شستا، خودرو، وبملت، خساپا، وغدیر، کگل، فملی، پارسان، و ...
        """
        query.edit_message_text(help_text, reply_markup=main_menu())

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text.strip()
    action = context.user_data.get('action', 'analyze')

    if action == 'analyze':
        parts = text.split()
        symbol = parts[0]
        update.message.reply_text(f"⏳ در حال تحلیل {symbol} ...")
        try:
            analyzer = AdvancedAnalyzer(symbol)
            if not analyzer.fetch_data():
                update.message.reply_text("❌ نماد معتبر نیست.")
                return
            if not analyzer.calculate_all_indicators():
                update.message.reply_text("❌ خطا در محاسبه شاخص‌ها.")
                return
            result = analyzer.get_full_analysis()
            update.message.reply_text(result, reply_markup=main_menu())
        except Exception as e:
            logger.error(f"خطا: {e}")
            update.message.reply_text("❌ خطایی رخ داد.")

    elif action == 'backtest':
        parts = text.split()
        if len(parts) < 1:
            update.message.reply_text("❌ لطفاً نماد را وارد کنید.\nمثال: `فولاد 100`")
            return
        symbol = parts[0]
        days = int(parts[1]) if len(parts) > 1 else 100

        update.message.reply_text(f"⏳ در حال بک‌تست {symbol} برای {days} روز ...")
        try:
            analyzer = AdvancedAnalyzer(symbol)
            if not analyzer.fetch_data(days=days+50):
                update.message.reply_text("❌ نماد معتبر نیست.")
                return
            if not analyzer.calculate_all_indicators():
                update.message.reply_text("❌ خطا در محاسبه شاخص‌ها.")
                return

            result = analyzer.backtest(days=days)
            if result is None:
                update.message.reply_text("❌ داده کافی برای بک‌تست وجود ندارد.")
                return

            text = f"""
📊 **نتایج بک‌تست {symbol}** ({days} روز)

💰 سرمایه اولیه: {result['initial_capital']:,.0f} تومان
💰 سرمایه نهایی: {result['final_capital']:,.0f} تومان
📈 بازدهی کل: {result['total_return']:+.2f}%
📊 تعداد معاملات: {result['num_trades']}
🏆 نرخ موفقیت: {result['win_rate']:.1f}%
📉 بازدهی خرید و نگهداری: {result['buy_hold_return']:+.2f}%
🎯 عملکرد نسبت به خرید و نگهداری: {result['total_return'] - result['buy_hold_return']:+.2f}%

📋 **جزئیات معاملات:**
{result['trades']}
"""
            update.message.reply_text(text, reply_markup=main_menu())
        except Exception as e:
            logger.error(f"خطا: {e}")
            update.message.reply_text("❌ خطایی رخ داد.")

    elif action == 'predict':
        parts = text.split()
        symbol = parts[0]
        update.message.reply_text(f"⏳ در حال پیش‌بینی {symbol} ...")
        try:
            analyzer = AdvancedAnalyzer(symbol)
            if not analyzer.fetch_data(days=200):
                update.message.reply_text("❌ نماد معتبر نیست.")
                return
            if not analyzer.calculate_all_indicators():
                update.message.reply_text("❌ خطا در محاسبه شاخص‌ها.")
                return

            pred = analyzer.predict_price(days_ahead=5)
            if pred is None:
                update.message.reply_text("❌ داده کافی برای پیش‌بینی وجود ندارد.")
                return

            text = f"🔮 **پیش‌بینی قیمت {symbol}** (۵ روز آینده)\n\n"
            text += f"قیمت فعلی: {pred['current_price']:,.0f} تومان\n━━━━━━━━━━━━━━━━━━\n"
            for i, price in enumerate(pred['predictions'], 1):
                text += f"روز {i}: {price:,.0f} تومان\n"
            if pred['trend']:
                text += "\n📈 **روند پیش‌بینی شده**:\n"
                for i, t in enumerate(pred['trend'], 1):
                    text += f"روز {i}→{i+1}: {t}\n"
            overall = "🟢 **روند کلی صعودی**" if pred['predictions'][-1] > pred['predictions'][0] else "🔴 **روند کلی نزولی**"
            text += f"\n🎯 {overall}"

            update.message.reply_text(text, reply_markup=main_menu())
        except Exception as e:
            logger.error(f"خطا: {e}")
            update.message.reply_text("❌ خطایی رخ داد.")

    elif action == 'signals':
        parts = text.split()
        symbol = parts[0]
        update.message.reply_text(f"⏳ در حال بررسی زمان معامله برای {symbol} ...")
        try:
            analyzer = AdvancedAnalyzer(symbol)
            if not analyzer.fetch_data(days=200):
                update.message.reply_text("❌ نماد معتبر نیست.")
                return
            if not analyzer.calculate_all_indicators():
                update.message.reply_text("❌ خطا در محاسبه شاخص‌ها.")
                return

            signals = analyzer.get_signals()
            if not signals:
                update.message.reply_text("⚠️ در بازه‌ی مورد بررسی سیگنال خاصی یافت نشد.")
                return

            text = f"⏰ **زمان‌های معامله برای {symbol}** (۱۰ سیگنال آخر)\n\n"
            for signal in signals:
                text += f"{signal[0]} در تاریخ {signal[1]} به قیمت {signal[2]:,.0f} تومان\n"

            update.message.reply_text(text, reply_markup=main_menu())
        except Exception as e:
            logger.error(f"خطا: {e}")
            update.message.reply_text("❌ خطایی رخ داد.")

    else:
        update.message.reply_text("❌ گزینه نامعتبر است. لطفاً از منوی اصلی استفاده کنید.", reply_markup=main_menu())

    # ریست action
    context.user_data['action'] = None

# ===================== اجرای اصلی ==========================

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button_handler))
    dp.add_handler(CommandHandler("analyze", handle_message))
    dp.add_handler(CommandHandler("backtest", handle_message))
    dp.add_handler(CommandHandler("predict", handle_message))
    dp.add_handler(CommandHandler("signals", handle_message))

    logger.info("🚀 ربات حرفه‌ای تحلیل بورس روشن شد!")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

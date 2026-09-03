# ============================================================
# ربات تحلیل بورس تهران - نسخه سازگار با python-telegram-bot 13.7
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import algotik_tse as tse
import ta
import warnings
warnings.filterwarnings('ignore')

# تنظیم لاگینگ
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ===================== دریافت توکن ============================
TOKEN = os.getenv('BOT_TOKEN')
if not TOKEN:
    raise ValueError("❌ توکن ربات یافت نشد! متغیر BOT_TOKEN را تنظیم کنید.")

# ===================== کلاس تحلیلگر ==========================
class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.df = None

    def fetch_data(self, days=200):
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

    def calculate_indicators(self):
        if self.df is None or self.df.empty:
            return False
        df = self.df.copy()
        close = df['Close'].values

        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

        volume = df['Volume'].values
        df['Volume_MA_20'] = pd.Series(volume).rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['SMA_20'] = pd.Series(close).rolling(window=20).mean()

        try:
            client = tse.get_client_type(symbol=self.symbol, include_jdate=True)
            if client is not None and not client.empty:
                df['Buy_Real'] = client['Buy_Real'].values if 'Buy_Real' in client else 0
                df['Sell_Real'] = client['Sell_Real'].values if 'Sell_Real' in client else 0
                df['Buy_Legal'] = client['Buy_Legal'].values if 'Buy_Legal' in client else 0
                df['Sell_Legal'] = client['Sell_Legal'].values if 'Sell_Legal' in client else 0
                df['Net_Real'] = df['Buy_Real'] - df['Sell_Real']
                df['Net_Legal'] = df['Buy_Legal'] - df['Sell_Legal']
                df['Net_Money'] = df['Net_Real'] + df['Net_Legal']
                df['Net_Money_MA_5'] = df['Net_Money'].rolling(window=5).mean()
        except:
            pass

        self.df = df
        return True

    def get_analysis(self):
        if self.df is None or self.df.empty:
            return "❌ داده‌ای برای تحلیل وجود ندارد."
        last = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) > 1 else last
        current_price = last['Close']
        prev_close = prev['Close']
        price_change = ((current_price - prev_close) / prev_close) * 100

        macd = last['MACD']
        macd_signal = last['MACD_Signal']
        if pd.isna(macd) or pd.isna(macd_signal):
            macd_status = "⚠️ داده کافی نیست"
            macd_signal_text = ""
        elif macd > macd_signal:
            macd_status = "✅ روند صعودی"
            macd_signal_text = "🔺 سیگنال خرید"
        else:
            macd_status = "❌ روند نزولی"
            macd_signal_text = "🔻 سیگنال فروش"

        rsi = last['RSI']
        if pd.isna(rsi):
            rsi_status = "⚠️ داده کافی نیست"
        elif rsi > 70:
            rsi_status = f"⚠️ اشباع خرید ({rsi:.1f})"
        elif rsi < 30:
            rsi_status = f"✅ اشباع فروش ({rsi:.1f})"
        else:
            rsi_status = f"⚖️ خنثی ({rsi:.1f})"

        volume = last['Volume']
        vol_ma_20 = last['Volume_MA_20']
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

        net_money = last.get('Net_Money', 0)
        net_real = last.get('Net_Real', 0)
        net_legal = last.get('Net_Legal', 0)
        if net_money != 0:
            money_status = f"💰 ورود پول" if net_money > 0 else f"💸 خروج پول"
            money_status += f" ({abs(net_money):,.0f} تومان)"
        else:
            money_status = "⚠️ در دسترس نیست"

        score = 0
        signals = []
        if not pd.isna(macd) and not pd.isna(macd_signal):
            score += 1 if macd > macd_signal else -1
            signals.append("MACD خرید" if macd > macd_signal else "MACD فروش")
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
        if net_money > 0:
            score += 1
            signals.append("ورود پول")
        elif net_money < 0:
            score -= 1
            signals.append("خروج پول")

        if score >= 2:
            overall = "🟢 **روند صعودی** - شرایط خرید مناسب است"
        elif score >= 0.5:
            overall = "🟡 **روند خنثی تا صعودی** - احتیاط"
        elif score >= -1:
            overall = "🟠 **روند خنثی تا نزولی** - ریسک بالا"
        else:
            overall = "🔴 **روند نزولی** - از خرید خودداری کنید"

        text = f"""
📊 **تحلیل سهم {self.symbol}**
📅 تاریخ: {last.get('JDate', 'نامشخص')}

━━━━━━━━━━━━━━━━━━━━━━━━━━
💹 **قیمت‌ها**
   قیمت پایانی: {current_price:,.0f} تومان
   تغییر روزانه: {price_change:+.2f}%
   بیشترین: {last['High']:,.0f} | کمترین: {last['Low']:,.0f}
   میانگین ۲۰ روزه: {last['SMA_20']:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **MACD** : {macd_status} {macd_signal_text}
📊 **RSI** : {rsi_status}
📊 **حجم** : {vol_status}
💰 **پول** : {money_status}
━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **جمع‌بندی نهایی** (امتیاز: {score:.1f}/4)
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
        buy_condition = (df['RSI'] < 30) & (df['MACD'] > df['MACD_Signal'])
        sell_condition = (df['RSI'] > 70) & (df['MACD'] < df['MACD_Signal'])
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
                trades.append(('buy', i, price))
            elif signals.iloc[i] == -1 and position == 1:
                price = df['Close'].iloc[i]
                capital += shares * price
                shares = 0
                position = 0
                trades.append(('sell', i, price))
        if position == 1:
            price = df['Close'].iloc[-1]
            capital += shares * price
            trades.append(('sell', len(df)-1, price))
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
        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'buy_hold_return': buy_hold_return
        }

    def predict_future(self, days_ahead=5):
        if self.df is None or len(self.df) < 30:
            return None
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
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
            trend.append('صعودی' if predictions[i] > predictions[i-1] else 'نزولی')
        return {
            'predictions': predictions,
            'trend': trend,
            'current_price': self.df['Close'].iloc[-1]
        }

# ===================== دستورات ربات =========================

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🤖 **ربات تحلیل بورس تهران**\n\n"
        "📌 دستورات:\n"
        "/start - راهنما\n"
        "/analyze <نماد> - تحلیل لحظه‌ای\n"
        "/backtest <نماد> <تعداد روز> - بک‌تست\n"
        "/predict <نماد> - پیش‌بینی ۵ روز آینده\n"
        "/help - راهنمای کامل\n\n"
        "مثال: `/analyze فولاد`"
    )

def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "📖 **راهنمای کامل**\n\n"
        "🔹 /analyze <نماد> : تحلیل تکنیکال با MACD, RSI, حجم و ورود/خروج پول\n"
        "🔹 /backtest <نماد> <تعداد روز> : تست استراتژی روی داده‌های گذشته\n"
        "🔹 /predict <نماد> : پیش‌بینی قیمت برای ۵ روز آینده\n\n"
        "📌 نمادهای معتبر: فولاد، شستا، خودرو، وبملت، خساپا، وغدیر و ..."
    )

def analyze(update: Update, context: CallbackContext):
    args = context.args
    if not args:
        update.message.reply_text("❌ لطفاً نماد را وارد کنید.\nمثال: `/analyze فولاد`")
        return
    symbol = args[0].strip()
    update.message.reply_text(f"⏳ در حال تحلیل {symbol} ...")
    try:
        analyzer = StockAnalyzer(symbol)
        if not analyzer.fetch_data():
            update.message.reply_text(f"❌ نماد '{symbol}' معتبر نیست.")
            return
        if not analyzer.calculate_indicators():
            update.message.reply_text("❌ خطا در محاسبه شاخص‌ها.")
            return
        result = analyzer.get_analysis()
        update.message.reply_text(result)
    except Exception as e:
        logger.error(f"خطا در تحلیل: {e}")
        update.message.reply_text("❌ خطایی رخ داد. مجدداً تلاش کنید.")

def backtest_command(update: Update, context: CallbackContext):
    args = context.args
    if len(args) < 1:
        update.message.reply_text("❌ لطفاً نماد را وارد کنید.\nمثال: `/backtest فولاد 100`")
        return
    symbol = args[0]
    days = 100
    if len(args) >= 2:
        try:
            days = int(args[1])
        except:
            pass
    update.message.reply_text(f"⏳ در حال بک‌تست {symbol} برای {days} روز ...")
    try:
        analyzer = StockAnalyzer(symbol)
        if not analyzer.fetch_data(days=days+50):
            update.message.reply_text("❌ نماد معتبر نیست.")
            return
        if not analyzer.calculate_indicators():
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
"""
        update.message.reply_text(text)
    except Exception as e:
        logger.error(f"خطا در بک‌تست: {e}")
        update.message.reply_text("❌ خطایی رخ داد.")

def predict_command(update: Update, context: CallbackContext):
    args = context.args
    if not args:
        update.message.reply_text("❌ لطفاً نماد را وارد کنید.\nمثال: `/predict فولاد`")
        return
    symbol = args[0]
    update.message.reply_text(f"⏳ در حال پیش‌بینی {symbol} ...")
    try:
        analyzer = StockAnalyzer(symbol)
        if not analyzer.fetch_data(days=150):
            update.message.reply_text("❌ نماد معتبر نیست.")
            return
        if not analyzer.calculate_indicators():
            update.message.reply_text("❌ خطا در محاسبه شاخص‌ها.")
            return
        pred = analyzer.predict_future()
        if pred is None:
            update.message.reply_text("❌ داده کافی برای پیش‌بینی وجود ندارد (نیاز به حداقل ۳۰ روز داده).")
            return
        text = f"🔮 **پیش‌بینی قیمت {symbol}** (۵ روز آینده)\n\nقیمت فعلی: {pred['current_price']:,.0f} تومان\n━━━━━━━━━━━━━━━━━━\n"
        for i, price in enumerate(pred['predictions'], 1):
            text += f"روز {i}: {price:,.0f} تومان\n"
        if pred['trend']:
            text += "\n📈 **روند پیش‌بینی شده**:\n"
            for i, t in enumerate(pred['trend'], 1):
                emoji = "🟢" if t == "صعودی" else "🔴"
                text += f"روز {i}→{i+1}: {emoji} {t}\n"
        overall = "🟢 **روند کلی صعودی**" if pred['predictions'][-1] > pred['predictions'][0] else "🔴 **روند کلی نزولی**"
        text += f"\n🎯 {overall}"
        update.message.reply_text(text)
    except Exception as e:
        logger.error(f"خطا در پیش‌بینی: {e}")
        update.message.reply_text("❌ خطایی رخ داد.")

# ===================== اجرای اصلی =========================

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher
    
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("analyze", analyze))
    dp.add_handler(CommandHandler("backtest", backtest_command))
    dp.add_handler(CommandHandler("predict", predict_command))
    
    logger.info("🚀 ربات تحلیل بورس شروع به کار کرد...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

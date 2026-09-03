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

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv('BOT_TOKEN')
if not TOKEN:
    raise ValueError("توکن ربات یافت نشد!")

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.df = None

    def fetch_data(self, days=200):
        try:
            self.df = tse.get_history(symbol=self.symbol, adjust=True, include_jdate=True)
            if self.df is None or self.df.empty:
                return False
            self.df = self.df.tail(days).sort_index()
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
        df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

        volume = df['Volume'].values
        df['Volume_MA_20'] = pd.Series(volume).rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['SMA_20'] = pd.Series(close).rolling(20).mean()

        try:
            client = tse.get_client_type(symbol=self.symbol, include_jdate=True)
            if client is not None and not client.empty:
                df['Buy_Real'] = client['Buy_Real'].values if 'Buy_Real' in client else 0
                df['Sell_Real'] = client['Sell_Real'].values if 'Sell_Real' in client else 0
                df['Net_Real'] = df['Buy_Real'] - df['Sell_Real']
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
        price_change = ((current_price - prev['Close']) / prev['Close']) * 100

        macd = last['MACD']
        macd_signal = last['MACD_Signal']
        if pd.isna(macd) or pd.isna(macd_signal):
            macd_status = "⚠️ داده کافی نیست"
        elif macd > macd_signal:
            macd_status = "✅ روند صعودی (سیگنال خرید)"
        else:
            macd_status = "❌ روند نزولی (سیگنال فروش)"

        rsi = last['RSI']
        if pd.isna(rsi):
            rsi_status = "⚠️ داده کافی نیست"
        elif rsi > 70:
            rsi_status = f"⚠️ اشباع خرید ({rsi:.1f})"
        elif rsi < 30:
            rsi_status = f"✅ اشباع فروش ({rsi:.1f})"
        else:
            rsi_status = f"⚖️ خنثی ({rsi:.1f})"

        vol_ratio = last['Volume_Ratio']
        if pd.isna(vol_ratio):
            vol_status = "⚠️ داده کافی نیست"
        elif vol_ratio > 1.5:
            vol_status = f"📈 بالا ({vol_ratio:.1f}x میانگین)"
        else:
            vol_status = f"📊 عادی ({vol_ratio:.1f}x)"

        net_real = last.get('Net_Real', 0)
        money_status = f"💰 ورود {net_real:,.0f}" if net_real > 0 else f"💸 خروج {abs(net_real):,.0f}" if net_real < 0 else "⚖️ متعادل"

        text = f"""
📊 **تحلیل {self.symbol}**
📅 {last.get('JDate', 'نامشخص')}
💰 قیمت: {current_price:,.0f} تومان | تغییر: {price_change:+.2f}%
📈 MACD: {macd_status}
📊 RSI: {rsi_status}
📊 حجم: {vol_status}
💰 پول حقیقی: {money_status}
"""
        return text

def start(update: Update, context: CallbackContext):
    update.message.reply_text("🤖 ربات تحلیل بورس\n/analyze <نماد> - تحلیل سهم")

def analyze(update: Update, context: CallbackContext):
    if not context.args:
        update.message.reply_text("مثال: /analyze فولاد")
        return
    symbol = context.args[0]
    update.message.reply_text(f"⏳ تحلیل {symbol}...")
    try:
        analyzer = StockAnalyzer(symbol)
        if not analyzer.fetch_data() or not analyzer.calculate_indicators():
            update.message.reply_text("❌ نماد معتبر نیست.")
            return
        update.message.reply_text(analyzer.get_analysis())
    except Exception as e:
        logger.error(f"خطا: {e}")
        update.message.reply_text("❌ خطا رخ داد.")

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("analyze", analyze))
    logger.info("ربات روشن شد ✅")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

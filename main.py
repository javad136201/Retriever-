# ============================================================
# ربات تحلیل بورس تهران - نسخه ساده و سازگار با Railway
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
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

        # MACD
        macd = ta.trend.MACD(close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # حجم معاملات
        volume = df['Volume'].values
        df['Volume_MA_20'] = pd.Series(volume).rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']

        # میانگین متحرک ساده
        df['SMA_20'] = pd.Series(close).rolling(window=20).mean()

        # ورود و خروج پول (حقیقی/حقوقی)
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

        # MACD
        macd = last['MACD']
        macd_signal = last['MACD_Signal']
        if pd.isna(macd) or pd.isna(macd_signal):
            macd_status = "⚠️ داده کافی نیست"
        elif macd > macd_signal:
            macd_status = "✅ روند صعودی (سیگنال خرید)"
        else:
            macd_status = "❌ روند نزولی (سیگنال فروش)"

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

        # حجم
        vol_ratio = last['Volume_Ratio']
        if pd.isna(vol_ratio):
            vol_status = "⚠️ داده کافی نیست"
        elif vol_ratio > 2:
            vol_status = f"🔥 بسیار بالا ({vol_ratio:.1f}x)"
        elif vol_ratio > 1.5:
            vol_status = f"📈 بالاتر از میانگین ({vol_ratio:.1f}x)"
        else:
            vol_status = f"📊 عادی ({vol_ratio:.1f}x)"

        # ورود/خروج پول
        net_money = last.get('Net_Money', 0)
        if net_money != 0:
            money_status = f"💰 ورود پول" if net_money > 0 else f"💸 خروج پول"
            money_status += f" ({abs(net_money):,.0f} تومان)"
        else:
            money_status = "⚠️ در دسترس نیست"

        # جمع‌بندی
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
📈 **MACD** : {macd_status}
📊 **RSI** : {rsi_status}
📊 **حجم** : {vol_status}
💰 **پول** : {money_status}
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return text

# ===================== دستورات ربات =========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 **ربات تحلیل بورس تهران**\n\n"
        "📌 دستورات:\n"
        "/start - راهنما\n"
        "/analyze <نماد> - تحلیل لحظه‌ای\n"
        "/help - راهنمای کامل\n\n"
        "مثال: `/analyze فولاد`"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 **راهنمای کامل**\n\n"
        "🔹 /analyze <نماد> : تحلیل تکنیکال با MACD, RSI, حجم و ورود/خروج پول\n"
        "📌 نمادهای معتبر: فولاد، شستا، خودرو، وبملت، خساپا، وغدیر و ..."
    )

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❌ لطفاً نماد را وارد کنید.\nمثال: `/analyze فولاد`")
        return

    symbol = context.args[0].strip()
    await update.message.reply_text(f"⏳ در حال تحلیل {symbol} ...")

    try:
        analyzer = StockAnalyzer(symbol)
        if not analyzer.fetch_data():
            await update.message.reply_text(f"❌ نماد '{symbol}' معتبر نیست.")
            return

        if not analyzer.calculate_indicators():
            await update.message.reply_text("❌ خطا در محاسبه شاخص‌ها.")
            return

        result = analyzer.get_analysis()
        await update.message.reply_text(result)

    except Exception as e:
        logger.error(f"خطا در تحلیل: {e}")
        await update.message.reply_text("❌ خطایی رخ داد. مجدداً تلاش کنید.")

# ===================== اجرای اصلی =========================

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("analyze", analyze))

    logger.info("🚀 ربات تحلیل بورس شروع به کار کرد...")
    app.run_polling()

if __name__ == "__main__":
    main()

# ============================================================
# ربات تحلیل بورس با پروکسی از فایل - نسخه Railway
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler
import algotik_tse as tse
import ta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = os.getenv('BOT_TOKEN')
if not TOKEN:
    raise ValueError("❌ توکن ربات یافت نشد!")

# ===================== بارگذاری پروکسی‌ها از فایل ==========================

def load_proxies_from_file():
    """بارگذاری لیست پروکسی‌ها از فایل proxies.txt"""
    proxies = []
    try:
        with open('proxies.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    proxies.append(f'http://{line}')
        logger.info(f"✅ {len(proxies)} پروکسی از فایل بارگذاری شد")
    except FileNotFoundError:
        logger.warning("⚠️ فایل proxies.txt پیدا نشد، از لیست پیش‌فرض استفاده می‌شود")
        # لیست پیش‌فرض (چند تا پروکسی معروف)
        proxies = [
            'http://185.162.231.73:80',
            'http://185.162.228.119:80',
            'http://15.160.116.45:39597',
        ]
    except Exception as e:
        logger.error(f"خطا در بارگذاری فایل پروکسی: {e}")
    return proxies

# بارگذاری پروکسی‌ها در حافظه (یک بار در ابتدا)
PROXY_LIST = load_proxies_from_file()

# ===================== کلاس تحلیلگر ==========================

class AdvancedAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.df = None
        self.error = None

    def fetch_data(self, days=200):
        """دریافت داده با استفاده از پروکسی‌های لیست"""
        
        # ۱. تلاش با پروکسی محیطی (اگر تنظیم شده باشه)
        env_proxy = os.getenv('HTTP_PROXY')
        if env_proxy:
            logger.info(f"🔄 استفاده از پروکسی محیطی: {env_proxy}")
            try:
                self.df = tse.get_history(
                    symbol=self.symbol, adjust=True, include_jdate=True,
                    proxies={'http': env_proxy, 'https': env_proxy}
                )
                if self.df is not None and not self.df.empty:
                    self.df = self.df.tail(days).sort_index()
                    return True
            except Exception as e:
                logger.warning(f"پروکسی محیطی کار نکرد: {e}")

        # ۲. تلاش مستقیم (بدون پروکسی)
        try:
            logger.info("🔄 تلاش برای اتصال مستقیم...")
            self.df = tse.get_history(symbol=self.symbol, adjust=True, include_jdate=True)
            if self.df is not None and not self.df.empty:
                self.df = self.df.tail(days).sort_index()
                logger.info("✅ اتصال مستقیم موفقیت‌آمیز بود")
                return True
        except Exception as e:
            logger.warning(f"اتصال مستقیم ناموفق: {e}")

        # ۳. تست پروکسی‌های موجود در لیست
        if not PROXY_LIST:
            self.error = "لیست پروکسی خالی است"
            return False

        logger.info(f"🔄 تست {len(PROXY_LIST)} پروکسی از لیست...")
        for proxy in PROXY_LIST:
            try:
                logger.info(f"🔄 تست پروکسی: {proxy}")
                self.df = tse.get_history(
                    symbol=self.symbol, adjust=True, include_jdate=True,
                    proxies={'http': proxy, 'https': proxy},
                    timeout=10  # ۱۰ ثانیه وقت بده
                )
                if self.df is not None and not self.df.empty:
                    self.df = self.df.tail(days).sort_index()
                    logger.info(f"✅ موفقیت با پروکسی: {proxy}")
                    return True
            except Exception as e:
                logger.warning(f"پروکسی {proxy} کار نکرد: {str(e)[:50]}")
                continue

        self.error = "همه پروکسی‌ها ناموفق بودند"
        logger.error(self.error)
        return False

    def calculate_indicators(self):
        if self.df is None or self.df.empty:
            return False
        try:
            df = self.df.copy()
            close = df['Close'].values
            volume = df['Volume'].values

            macd = ta.trend.MACD(close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
            df['SMA_20'] = pd.Series(close).rolling(20).mean()
            df['Volume_MA_20'] = pd.Series(volume).rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']

            try:
                client = tse.get_client_type(symbol=self.symbol, include_jdate=True)
                if client is not None and not client.empty:
                    df['Net_Real'] = client['Buy_Real'].values - client['Sell_Real'].values
                else:
                    df['Net_Real'] = 0
            except:
                df['Net_Real'] = 0

            self.df = df
            return True
        except Exception as e:
            logger.error(f"خطا در محاسبه شاخص‌ها: {e}")
            return False

    def get_analysis(self):
        if self.df is None or self.df.empty:
            return f"❌ {self.error or 'داده‌ای وجود ندارد'}"

        try:
            last = self.df.iloc[-1]
            prev = self.df.iloc[-2] if len(self.df) > 1 else last

            current_price = last['Close']
            price_change = ((current_price - prev['Close']) / prev['Close']) * 100

            rsi = last['RSI']
            if pd.isna(rsi):
                rsi_status = "⚠️ داده کافی نیست"
            elif rsi > 70:
                rsi_status = f"⚠️ اشباع خرید ({rsi:.1f})"
            elif rsi < 30:
                rsi_status = f"✅ اشباع فروش ({rsi:.1f})"
            else:
                rsi_status = f"⚖️ خنثی ({rsi:.1f})"

            macd = last['MACD']
            macd_signal = last['MACD_Signal']
            if pd.isna(macd) or pd.isna(macd_signal):
                macd_status = "⚠️ داده کافی نیست"
            elif macd > macd_signal:
                macd_status = "✅ صعودی (سیگنال خرید)"
            else:
                macd_status = "❌ نزولی (سیگنال فروش)"

            vol_ratio = last['Volume_Ratio']
            if pd.isna(vol_ratio):
                vol_status = "⚠️ داده کافی نیست"
            elif vol_ratio > 1.5:
                vol_status = f"📈 بالا ({vol_ratio:.1f}x)"
            else:
                vol_status = f"📊 عادی ({vol_ratio:.1f}x)"

            net_real = last.get('Net_Real', 0)
            money_status = f"💰 ورود پول" if net_real > 0 else f"💸 خروج پول" if net_real < 0 else "⚖️ متعادل"
            if net_real != 0:
                money_status += f" ({abs(net_real):,.0f} تومان)"

            text = f"""
📊 **تحلیل {self.symbol}**
📅 {last.get('JDate', 'نامشخص')}

💰 قیمت: {current_price:,.0f} تومان | تغییر: {price_change:+.2f}%
📈 RSI: {rsi_status}
📈 MACD: {macd_status}
📊 حجم: {vol_status}
💰 پول حقیقی: {money_status}
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            return text

        except Exception as e:
            logger.error(f"خطا در ساخت تحلیل: {e}")
            return f"❌ خطا: {str(e)}"

# ===================== منو و دستورات ==========================

def main_menu():
    keyboard = [
        [InlineKeyboardButton("📊 تحلیل لحظه‌ای", callback_data='analyze')],
        [InlineKeyboardButton("📈 بک‌تست", callback_data='backtest')],
        [InlineKeyboardButton("📋 راهنما", callback_data='help')]
    ]
    return InlineKeyboardMarkup(keyboard)

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🤖 **ربات تحلیل بورس**\n\n"
        "لطفاً یکی از گزینه‌ها رو انتخاب کن:",
        reply_markup=main_menu()
    )

def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    if query.data == 'analyze':
        query.edit_message_text("🔍 لطفاً نماد رو وارد کن:\nمثال: `فولاد`")
        context.user_data['action'] = 'analyze'
    elif query.data == 'backtest':
        query.edit_message_text("📈 لطفاً نماد و تعداد روز رو وارد کن:\nمثال: `فولاد 100`")
        context.user_data['action'] = 'backtest'
    elif query.data == 'help':
        query.edit_message_text(
            "📖 **راهنما**\n\n"
            "/analyze <نماد> - تحلیل لحظه‌ای\n"
            "/backtest <نماد> <روز> - بک‌تست\n\n"
            "مثال: /analyze فولاد",
            reply_markup=main_menu()
        )

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text.strip()
    action = context.user_data.get('action', 'analyze')

    if not text:
        update.message.reply_text("❌ لطفاً یک نماد وارد کن.", reply_markup=main_menu())
        return

    parts = text.split()
    symbol = parts[0]
    days = int(parts[1]) if len(parts) > 1 else 100

    if action == 'analyze':
        update.message.reply_text(f"⏳ در حال تحلیل {symbol} ... (تست {len(PROXY_LIST)} پروکسی)")
        try:
            analyzer = AdvancedAnalyzer(symbol)
            if not analyzer.fetch_data():
                update.message.reply_text(f"❌ {analyzer.error}")
                return
            if not analyzer.calculate_indicators():
                update.message.reply_text("❌ خطا در محاسبه شاخص‌ها")
                return
            result = analyzer.get_analysis()
            update.message.reply_text(result, reply_markup=main_menu())
        except Exception as e:
            logger.error(f"خطا: {e}")
            update.message.reply_text(f"❌ خطا: {str(e)}", reply_markup=main_menu())

    elif action == 'backtest':
        update.message.reply_text(f"⏳ در حال بک‌تست {symbol} ...")
        update.message.reply_text("⚠️ این بخش در حال توسعه است", reply_markup=main_menu())

    context.user_data['action'] = None

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button_handler))
    dp.add_handler(CommandHandler("analyze", handle_message))
    dp.add_handler(CommandHandler("backtest", handle_message))

    logger.info(f"🚀 ربات با {len(PROXY_LIST)} پروکسی روشن شد!")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

# ============================================================
# ربات تحلیل بورس با tse-data و پروکسی - نسخه نهایی
# ============================================================

import os
import logging
import pandas as pd
import numpy as np
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler
import ta
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

# ===================== بارگذاری پروکسی‌ها ==========================

def load_proxies():
    """بارگذاری پروکسی از فایل یا لیست پیش‌فرض"""
    proxies = []
    try:
        with open('proxies.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    proxies.append(line)
        logger.info(f"✅ {len(proxies)} پروکسی از فایل بارگذاری شد")
    except:
        logger.warning("⚠️ فایل proxies.txt پیدا نشد، از لیست داخلی استفاده می‌شود")
        proxies = [
            '185.162.231.73:80',
            '185.162.228.119:80',
            '15.160.116.45:39597',
        ]
    return proxies

PROXY_LIST = load_proxies()

# ===================== دریافت داده با tse-data ==========================

def get_stock_data(symbol, days=200):
    """دریافت داده از بورس با tse-data و پروکسی"""
    
    # تلاش برای نصب tse-data در صورت نبود
    try:
        import tse_data
    except ImportError:
        logger.warning("⚠️ tse-data نصب نیست، در حال نصب...")
        os.system("pip install tse-data")
        import tse_data
    
    import tse_data
    
    # ۱. تلاش مستقیم (بدون پروکسی)
    try:
        logger.info("🔄 تلاش مستقیم...")
        df = tse_data.get_data(symbol, period='1y')
        if df is not None and not df.empty:
            df = df.tail(days)
            logger.info("✅ داده مستقیم دریافت شد")
            return df
    except Exception as e:
        logger.warning(f"اتصال مستقیم ناموفق: {e}")
    
    # ۲. تلاش با پروکسی‌ها
    for proxy in PROXY_LIST:
        try:
            logger.info(f"🔄 تست پروکسی: {proxy}")
            # تنظیم پروکسی برای requests
            proxies = {
                'http': f'http://{proxy}',
                'https': f'http://{proxy}'
            }
            # tse-data از requests استفاده می‌کند، پس با تنظیم session می‌توان پروکسی داد
            session = requests.Session()
            session.proxies.update(proxies)
            session.timeout = 10
            
            df = tse_data.get_data(symbol, period='1y', session=session)
            if df is not None and not df.empty:
                df = df.tail(days)
                logger.info(f"✅ داده با پروکسی {proxy} دریافت شد")
                return df
        except Exception as e:
            logger.warning(f"پروکسی {proxy} کار نکرد: {str(e)[:50]}")
            continue
    
    logger.error("❌ همه روش‌ها ناموفق بودند")
    return None

# ===================== کلاس تحلیلگر ==========================

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.df = None
        self.error = None

    def fetch_data(self, days=200):
        """دریافت داده با tse-data"""
        try:
            self.df = get_stock_data(self.symbol, days)
            if self.df is None or self.df.empty:
                self.error = "داده‌ای دریافت نشد"
                return False
            
            # اطمینان از وجود ستون‌های مورد نیاز
            required_cols = ['Close', 'High', 'Low', 'Volume']
            for col in required_cols:
                if col not in self.df.columns:
                    self.error = f"ستون {col} در داده وجود ندارد"
                    return False
            
            self.df = self.df.sort_index()
            return True
            
        except Exception as e:
            self.error = f"خطا: {str(e)}"
            logger.error(self.error)
            return False

    def calculate_indicators(self):
        if self.df is None or self.df.empty:
            return False
        
        try:
            df = self.df.copy()
            close = df['Close'].values
            volume = df['Volume'].values

            # MACD
            macd = ta.trend.MACD(close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(close, window=14).rsi()
            
            # SMA
            df['SMA_20'] = pd.Series(close).rolling(20).mean()
            df['SMA_50'] = pd.Series(close).rolling(50).mean()
            
            # حجم
            df['Volume_MA_20'] = pd.Series(volume).rolling(20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
            
            self.df = df
            return True
            
        except Exception as e:
            self.error = f"خطا در محاسبه شاخص‌ها: {e}"
            logger.error(self.error)
            return False

    def get_analysis(self):
        if self.df is None or self.df.empty:
            return f"❌ {self.error or 'داده‌ای وجود ندارد'}"

        try:
            last = self.df.iloc[-1]
            prev = self.df.iloc[-2] if len(self.df) > 1 else last

            current_price = last['Close']
            price_change = ((current_price - prev['Close']) / prev['Close']) * 100

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

            # MACD
            macd = last['MACD']
            macd_signal = last['MACD_Signal']
            if pd.isna(macd) or pd.isna(macd_signal):
                macd_status = "⚠️ داده کافی نیست"
            elif macd > macd_signal:
                macd_status = "✅ صعودی (سیگنال خرید)"
            else:
                macd_status = "❌ نزولی (سیگنال فروش)"

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

            text = f"""
📊 **تحلیل {self.symbol}**
📅 {last.get('JDate', 'نامشخص')}

━━━━━━━━━━━━━━━━━━━━━━━━━━
💹 **قیمت‌ها**
   قیمت پایانی: {current_price:,.0f} تومان
   تغییر روزانه: {price_change:+.2f}%
   بیشترین: {last['High']:,.0f} | کمترین: {last['Low']:,.0f}

━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **شاخص‌های تکنیکال**
   RSI: {rsi_status}
   MACD: {macd_status}
   میانگین متحرک: {ma_status}

━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **حجم معاملات**
   {vol_status}
   حجم امروز: {last['Volume']:,.0f}
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
        [InlineKeyboardButton("📋 راهنما", callback_data='help')]
    ]
    return InlineKeyboardMarkup(keyboard)

def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "🤖 **ربات تحلیل بورس**\n\n"
        "لطفاً یکی از گزینه‌ها رو انتخاب کنید:",
        reply_markup=main_menu()
    )

def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    query.answer()

    if query.data == 'analyze':
        query.edit_message_text("🔍 لطفاً نماد رو وارد کنید:\nمثال: `فولاد`")
        context.user_data['action'] = 'analyze'
    elif query.data == 'help':
        query.edit_message_text(
            "📖 **راهنما**\n\n"
            "/analyze <نماد> - تحلیل لحظه‌ای\n\n"
            "مثال: /analyze فولاد\n\n"
            f"📌 {len(PROXY_LIST)} پروکسی در لیست موجود است",
            reply_markup=main_menu()
        )

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text.strip()
    action = context.user_data.get('action', 'analyze')

    if not text:
        update.message.reply_text("❌ لطفاً یک نماد وارد کنید.", reply_markup=main_menu())
        return

    parts = text.split()
    symbol = parts[0]

    if action == 'analyze':
        update.message.reply_text(f"⏳ در حال تحلیل {symbol} ... (تست {len(PROXY_LIST)} پروکسی)")
        try:
            analyzer = StockAnalyzer(symbol)
            if not analyzer.fetch_data():
                update.message.reply_text(f"❌ {analyzer.error}\n\n💡 راهنما:\n- نماد را به فارسی وارد کنید\n- از نمادهای معتبر استفاده کنید: فولاد، شستا، خودرو، وبملت")
                return
            if not analyzer.calculate_indicators():
                update.message.reply_text(f"❌ {analyzer.error}")
                return
            result = analyzer.get_analysis()
            update.message.reply_text(result, reply_markup=main_menu())
        except Exception as e:
            logger.error(f"خطا: {e}")
            update.message.reply_text(f"❌ خطا: {str(e)}", reply_markup=main_menu())

    context.user_data['action'] = None

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button_handler))
    dp.add_handler(CommandHandler("analyze", handle_message))

    logger.info(f"🚀 ربات با {len(PROXY_LIST)} پروکسی روشن شد!")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()

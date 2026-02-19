from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(os.path.join(BASE_DIR, '.env'))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('SECRET_KEY', 'django-insecure-change-this-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
# When deploying to EC2, ensure DEBUG=False is set in your .env file

ALLOWED_HOSTS = os.environ.get(
    'ALLOWED_HOSTS',
    '54.162.20.141,thebakestory.store,www.thebakestory.store,localhost,127.0.0.1'
).split(',')

# Security settings for production
if not DEBUG:
    SECURE_SSL_REDIRECT = False  # Set to True once HTTPS/SSL is configured
    SESSION_COOKIE_SECURE = False  # Set to True with HTTPS
    CSRF_COOKIE_SECURE = False    # Set to True with HTTPS
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
    SECURE_HSTS_SECONDS = 0       # Enable after HTTPS is confirmed working

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'bakery',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # WhiteNoise for static files
    'corsheaders.middleware.CorsMiddleware',       # Must be before CommonMiddleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'bakery_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'bakery_project.wsgi.application'

# Database — SQLite (good for single-server EC2 deploy)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

# Static files configuration
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_DIRS = []

# WhiteNoise configuration for static files — only in production
if not DEBUG:
    STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Media files (user uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'

# ─── Razorpay ────────────────────────────────────────────────────────────────
RAZORPAY_KEY_ID = os.environ.get('RAZORPAY_KEY_ID', '')
RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET', '')

# ─── SMS Settings ─────────────────────────────────────────────────────────────
ADMIN_PHONE_NUMBER = '+918074691873'
BAKERY_NAME = 'The Bake Story'
SMS_NOTIFICATIONS_ENABLED = False

# ─── Email Configuration ──────────────────────────────────────────────────────
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
EMAIL_HOST_USER = os.environ.get('ADMIN_EMAIL', 'btechmuthyam@gmail.com')
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'btechmuthyam@gmail.com')
EMAIL_NOTIFICATIONS_ENABLED = False

# ─── Order Notification Settings ──────────────────────────────────────────────
ORDER_NOTIFICATION_EMAIL = os.environ.get('ADMIN_EMAIL', 'btechmuthyam@gmail.com')
ORDER_EMAIL_NOTIFICATIONS_ENABLED = False
ORDER_SMS_NOTIFICATIONS_ENABLED = False
BAKERY_BUSINESS_NAME = 'The Bake Story'
BAKERY_BUSINESS_PHONE = '8074691873'
BAKERY_BUSINESS_ADDRESS = 'Chaitanyapuri, Dilsukhnagar, Hyderabad'
BAKERY_BUSINESS_EMAIL = os.environ.get('ADMIN_EMAIL', 'btechmuthyam@gmail.com')

# ─── CORS Settings ────────────────────────────────────────────────────────────
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://54.162.20.141",
    "http://thebakestory.store",
    "https://thebakestory.store",
    "http://www.thebakestory.store",
    "https://www.thebakestory.store",
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# CSRF trusted origins for production
CSRF_TRUSTED_ORIGINS = [
    "http://54.162.20.141",
    "http://thebakestory.store",
    "https://thebakestory.store",
    "http://www.thebakestory.store",
    "https://www.thebakestory.store",
]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'WARNING',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'django.log'),
            'formatter': 'verbose',
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'WARNING' if not DEBUG else 'DEBUG',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
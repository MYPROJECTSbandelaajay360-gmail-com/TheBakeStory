from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
from django.db.models import Prefetch, Sum, Count, Q
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.core.mail import send_mail
from .models import MenuItem, Order, OrderItem, Payment, UserProfile, Table
import json
import uuid
import razorpay
from django.utils import timezone
from decimal import Decimal
import os
from datetime import datetime, timedelta



def process_cart_items(cart_data):
    """Process cart data and return valid order items with total amount"""
    cart = json.loads(cart_data)
    if not cart:
        return None, None, "Cart is empty"
    
    total_amount = Decimal('0.00')
    order_items = []
    
    for item_id, item_data in cart.items():
        try:
            menu_item = MenuItem.objects.get(id=int(item_id))
            quantity = int(item_data.get('quantity', 0))
            if quantity > 0:
                subtotal = menu_item.price * quantity
                total_amount += subtotal
                order_items.append({
                    'menu_item': menu_item,
                    'quantity': quantity,
                    'price': menu_item.price
                })
        except (MenuItem.DoesNotExist, ValueError):
            continue
    
    if not order_items:
        return None, None, "No valid items in cart"
    
    return order_items, total_amount, None


# Simple template views
def index(request):
    return render(request, 'bakery/index.html')


def chatbot_view(request):
    """Display the chatbot interface"""
    return render(request, 'bakery/chatbot_premium_v2.html', {
        'table_number': ''
    })


def qr_landing_view(request):
    """Landing page after scanning QR code"""
    table_number = request.GET.get('table', '')
    return render(request, 'bakery/qr_landing.html', {
        'table_number': table_number
    })


def chatbot_order_view(request):
    """Chatbot ordering interface"""
    table_number = request.GET.get('table', '')
    return render(request, 'bakery/chatbot_premium_v2.html', {
        'table_number': table_number
    })


def admin_dashboard_view(request):
    """Admin dashboard for managing orders"""
    return render(request, 'bakery/admin_dashboard_v2.html')


def admin_stats_api(request):
    """API endpoint for admin dashboard stats"""
    from django.utils import timezone
    from django.db.models import Q
    from datetime import timedelta
    
    today = timezone.now().date()
    yesterday = today - timedelta(days=1)
    
    # Today's orders
    today_orders = Order.objects.filter(created_at__date=today)
    today_completed = today_orders.filter(status='completed')
    today_revenue = today_orders.filter(status='completed').aggregate(Sum('total_amount'))['total_amount__sum'] or 0
    
    # Yesterday's orders
    yesterday_orders = Order.objects.filter(created_at__date=yesterday)
    yesterday_revenue = yesterday_orders.filter(status='completed').aggregate(Sum('total_amount'))['total_amount__sum'] or 0
    
    # Active orders (not completed/cancelled)
    active_orders = Order.objects.filter(
        ~Q(status__in=['completed', 'cancelled', 'served'])
    ).order_by('-created_at')
    
    # Calculate revenue change percentage
    revenue_change = 0
    if yesterday_revenue > 0:
        revenue_change = ((float(today_revenue) - float(yesterday_revenue)) / float(yesterday_revenue)) * 100
    elif today_revenue > 0:
        revenue_change = 100
    
    # Calculate average preparation time
    completed_with_times = Order.objects.filter(
        status='completed',
        created_at__date=today,
        ready_at__isnull=False
    )
    
    if completed_with_times.exists():
        total_time = sum([(order.ready_at - order.created_at).total_seconds() for order in completed_with_times])
        avg_seconds = total_time / completed_with_times.count()
        avg_minutes = int(avg_seconds / 60)
        avg_time = f"{avg_minutes}m"
    else:
        avg_time = "--"
    
    # Prepare orders data
    orders_data = []
    for order in active_orders[:10]:  # Last 10 active orders
        orders_data.append({
            'order_id': order.order_id,
            'customer': order.customer_name or (order.user.get_full_name() if order.user else 'Guest'),
            'status': order.status,
            'amount': str(order.total_amount),
            'type': order.get_order_type_display(),
            'table': str(order.table.table_number) if order.table else 'N/A',
            'time': order.created_at.strftime('%H:%M'),
        })
    
    return JsonResponse({
        'today_revenue': float(today_revenue),
        'yesterday_revenue': float(yesterday_revenue),
        'revenue_change': revenue_change,
        'active_orders_count': active_orders.count(),
        'completed_today': today_completed.count(),
        'total_today': today_orders.count(),
        'avg_time': avg_time,
        'orders': orders_data
    })


def kitchen_portal_view(request):
    """Kitchen staff portal for managing orders"""
    return render(request, 'bakery/kitchen_portal_v2.html')


def menu_view(request):
    menu_items = MenuItem.objects.filter(available=True)
    table_number = request.GET.get('table', '')
    mode = request.GET.get('mode', 'browse')  # browse or chatbot
    
    return render(request, 'bakery/menu.html', {
        'menu_items': menu_items,
        'table_number': table_number,
        'mode': mode
    })


def about_view(request):
    return render(request, 'bakery/about.html')


def contact_view(request):
    return render(request, 'bakery/contact.html')



@login_required
def cart_view(request):
    return render(request, 'bakery/cart.html')


@login_required
def orders_view(request):
    """Display user-specific orders - only orders belonging to the logged-in user"""
    if not request.user.is_authenticated:
        return redirect('login')
    
    # Optimized query with prefetch to reduce database hits
    orders_queryset = Order.objects.filter(user=request.user).select_related(
        'payment'
    ).prefetch_related(
        'items__menu_item'
    ).order_by('-created_at')
    
    # Get current active orders
    current_orders = orders_queryset.filter(status__in=['pending', 'confirmed', 'preparing', 'ready'])
    
    # Get order history
    order_history = orders_queryset.filter(status__in=['completed', 'cancelled'])
    
    # Calculate statistics
    total_orders = orders_queryset.count()
    completed_orders = orders_queryset.filter(status='completed').count()
    
    context = {
        'current_orders': current_orders,
        'order_history': order_history,
        'total_orders': total_orders,
        'completed_orders': completed_orders,
        'user_full_name': request.user.get_full_name() or request.user.username
    }
    return render(request, 'bakery/orders.html', context)


@login_required
def order_detail_view(request, order_id):
    """Display full detail of a single order belonging to the logged-in user"""
    from django.shortcuts import get_object_or_404
    order = get_object_or_404(Order, order_id=order_id, user=request.user)
    return render(request, 'bakery/order_detail.html', {'order': order})



@login_required
def upi_payment_view(request):
    if request.method != 'POST':
        return render(request, 'bakery/upi-payment.html')
    
    # Get POST data
    cart_data = request.POST.get('cart_data')
    delivery_address = request.POST.get('delivery_address')
    delivery_phone = request.POST.get('delivery_phone')
    delivery_notes = request.POST.get('delivery_notes', '')
    payment_screenshot = request.FILES.get('payment_screenshot')
    upi_transaction_id = request.POST.get('upi_transaction_id', '')
    
    # Validate required fields
    if not all([cart_data, delivery_address, payment_screenshot]):
        messages.error(request, 'Please fill all required fields and upload payment screenshot.')
        return render(request, 'bakery/upi-payment.html')
    
    try:
        # Process cart items
        order_items, total_amount, error = process_cart_items(cart_data)
        if error:
            messages.error(request, error)
            return redirect('cart')
        
        # Create order
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        order = Order.objects.create(
            user=request.user,
            order_id=order_id,
            status='pending',
            total_amount=total_amount,
            delivery_address=delivery_address,
            delivery_phone=delivery_phone,
            delivery_notes=delivery_notes
        )
        
        # Create order items
        OrderItem.objects.bulk_create([
            OrderItem(
                order=order,
                menu_item=item['menu_item'],
                quantity=item['quantity'],
                price=item['price']
            ) for item in order_items
        ])
        
        # Create payment record
        transaction_id = f"TXN-{uuid.uuid4().hex[:12].upper()}"
        Payment.objects.create(
            order=order,
            payment_method='upi',
            payment_status='pending',
            transaction_id=transaction_id,
            amount=order.grand_total,
            upi_id=upi_transaction_id,
            payment_screenshot=payment_screenshot
        )
        
        # Send order notifications (Email + SMS)
        print(f"📦 Sending notifications for Order #{order.id}...")
        send_order_notification_email(order)
        send_order_sms_notification(order)
        
        success_message = f'Order placed successfully! Order ID: {order_id}. Your payment will be verified within 24 hours.'
        messages.success(request, success_message)
        
        return render(request, 'bakery/upi-payment.html', {
            'order_placed': True,
            'order_id': order_id,
            'message': success_message
        })
        
    except json.JSONDecodeError:
        messages.error(request, 'Invalid cart data.')
    except Exception as e:
        messages.error(request, f'Error processing order: {str(e)}')
    
    return render(request, 'bakery/upi-payment.html')


# Authentication views
def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        
        if user:
            login(request, user)
            messages.success(request, 'Login successful!')
            return redirect(request.GET.get('next', 'index'))
        else:
            messages.error(request, 'Invalid email or password')
    
    return render(request, 'bakery/login.html')


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        fullname = request.POST.get('fullname', '')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        if User.objects.filter(username=email).exists():
            messages.error(request, 'Email already registered')
            return render(request, 'bakery/signin.html')
        
        # Parse fullname
        name_parts = fullname.split() if fullname else []
        first_name = name_parts[0] if name_parts else ''
        last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
        
        # Create user
        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name
        )
        login(request, user)
        messages.success(request, 'Account created successfully!')
        return redirect('index')
    
    return render(request, 'bakery/signin.html')


def logout_view(request):
    logout(request)
    messages.success(request, 'Logged out successfully!')
    return redirect('index')


def forgot_password_view(request):
    """Show forgot password form and send reset email"""
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        email = request.POST.get('email', '').strip()
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            messages.error(request, 'No account found with that email address.')
            return render(request, 'bakery/forgot_password.html')

        # Generate secure token
        from django.contrib.auth.tokens import default_token_generator
        from django.utils.http import urlsafe_base64_encode
        from django.utils.encoding import force_bytes

        uid = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)

        # Build reset link
        reset_url = request.build_absolute_uri(
            f'/reset-password/{uid}/{token}/'
        )

        # Send reset email
        user_name = user.get_full_name() or user.email.split('@')[0]
        html_message = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 500px; margin: 0 auto; background: #fffaf5; border-radius: 16px; overflow: hidden; border: 1px solid #f0e0cc;">
            <div style="background: linear-gradient(135deg, #d2691e 0%, #b8571e 100%); padding: 30px; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 24px;">🍰 The Bake Story</h1>
                <p style="color: rgba(255,255,255,0.9); margin-top: 5px; font-size: 14px;">Password Reset Request</p>
            </div>
            <div style="padding: 30px;">
                <p style="color: #333; font-size: 16px;">Hi <strong>{user_name}</strong>,</p>
                <p style="color: #555; font-size: 15px; line-height: 1.6;">We received a request to reset your password. Click the button below to set a new password:</p>
                <div style="text-align: center; margin: 25px 0;">
                    <a href="{reset_url}" style="background: linear-gradient(135deg, #d2691e 0%, #b8571e 100%); color: white; padding: 14px 40px; border-radius: 10px; text-decoration: none; font-weight: 700; font-size: 16px; display: inline-block;">Reset Password</a>
                </div>
                <p style="color: #888; font-size: 13px; line-height: 1.5;">If you didn't request this, you can safely ignore this email. This link will expire automatically.</p>
                <hr style="border: none; border-top: 1px solid #f0e0cc; margin: 20px 0;">
                <p style="color: #aaa; font-size: 12px; text-align: center;">The Bake Story &bull; Chaitanyapuri, Dilsukhnagar, Hyderabad</p>
            </div>
        </div>
        """
        try:
            from django.core.mail import send_mail
            send_mail(
                subject='Reset Your Password - The Bake Story',
                message=f'Hi {user_name}, reset your password here: {reset_url}',
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                html_message=html_message,
                fail_silently=False,
            )
            messages.success(request, 'A password reset link has been sent to your email!')
        except Exception as e:
            print(f"❌ Email send failed: {e}")
            messages.error(request, 'Failed to send reset email. Please try again later.')

        return render(request, 'bakery/forgot_password.html')

    return render(request, 'bakery/forgot_password.html')


def reset_password_view(request, uidb64, token):
    """Validate token and allow user to set new password"""
    from django.contrib.auth.tokens import default_token_generator
    from django.utils.http import urlsafe_base64_decode

    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is None or not default_token_generator.check_token(user, token):
        messages.error(request, 'This password reset link is invalid or has expired.')
        return redirect('forgot_password')

    if request.method == 'POST':
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')

        if len(password) < 6:
            messages.error(request, 'Password must be at least 6 characters long.')
            return render(request, 'bakery/reset_password.html', {'valid_link': True})

        if password != confirm_password:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'bakery/reset_password.html', {'valid_link': True})

        # set_password handles PBKDF2 hashing + salting automatically
        user.set_password(password)
        user.save()
        messages.success(request, 'Your password has been reset successfully! Please login with your new password.')
        return redirect('login')

    return render(request, 'bakery/reset_password.html', {'valid_link': True})


# Razorpay Integration - Initialize client
razorpay_client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))


@login_required
def payment_view(request):
    if request.method == 'POST':
        cart_data = request.POST.get('cart_data')
        delivery_address = request.POST.get('delivery_address')
        delivery_phone = request.POST.get('delivery_phone')
        delivery_notes = request.POST.get('delivery_notes', '')
        order_type = request.POST.get('order_type', 'delivery')

        # Validate required fields
        if not all([cart_data, delivery_address, delivery_phone]):
            messages.error(request, 'Please fill all required fields.')
            return redirect('cart')

        try:
            # Process cart items
            order_items, total_amount, error = process_cart_items(cart_data)
            if error:
                messages.error(request, error)
                return redirect('cart')

            # Determine delivery fee
            delivery_fee = Decimal('50.00') if order_type == 'delivery' else Decimal('0.00')

            # Create order in database
            order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
            order = Order.objects.create(
                user=request.user,
                order_id=order_id,
                order_type=order_type,
                status='pending',
                total_amount=total_amount,
                delivery_fee=delivery_fee,
                delivery_address=delivery_address,
                delivery_phone=delivery_phone,
                delivery_notes=delivery_notes
            )

            # Create order items
            OrderItem.objects.bulk_create([
                OrderItem(
                    order=order,
                    menu_item=item['menu_item'],
                    quantity=item['quantity'],
                    price=item['price']
                ) for item in order_items
            ])

            # Create Razorpay order (amount in paise: ₹100 = 10000 paise)
            grand_total = Decimal(str(order.total_amount)) + Decimal(str(order.delivery_fee))
            razorpay_amount = int(float(grand_total) * 100)

            razorpay_order = razorpay_client.order.create({
                'amount': razorpay_amount,
                'currency': 'INR',
                'receipt': order_id,
                'payment_capture': '1'
            })

            # Store Razorpay order ID
            order.razorpay_order_id = razorpay_order['id']
            order.save()

            context = {
                'order': order,
                'razorpay_order_id': razorpay_order['id'],
                'razorpay_key_id': settings.RAZORPAY_KEY_ID,
                'amount': razorpay_amount,
                'currency': 'INR',
                'user_name': request.user.get_full_name() or request.user.username,
                'user_email': request.user.email,
                'user_phone': delivery_phone
            }
            return render(request, 'bakery/razorpay-payment.html', context)

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f'Payment view error: {str(e)}', exc_info=True)
            messages.error(request, 'Error creating order. Please try again.')
            return redirect('cart')

    return render(request, 'bakery/payment.html')


@csrf_exempt
@login_required
def razorpay_callback(request):
    """Handle Razorpay payment verification"""
    if request.method == 'POST':
        try:
            # Get payment details
            payment_id = request.POST.get('razorpay_payment_id')
            order_id = request.POST.get('razorpay_order_id')
            signature = request.POST.get('razorpay_signature')
            
            # Verify signature
            params_dict = {
                'razorpay_order_id': order_id,
                'razorpay_payment_id': payment_id,
                'razorpay_signature': signature
            }
            
            try:
                razorpay_client.utility.verify_payment_signature(params_dict)
                
                # Payment verified successfully
                order = Order.objects.get(razorpay_order_id=order_id)
                order.status = 'confirmed'
                order.save()
                
                # Create payment record
                Payment.objects.create(
                    order=order,
                    payment_method='razorpay',
                    payment_status='completed',
                    transaction_id=payment_id,
                    amount=order.grand_total
                )
                
                # Send order notifications (Email + SMS) after successful payment
                print(f"💳 Payment verified! Sending notifications for Order #{order.id}...")
                send_order_notification_email(order)
                send_order_sms_notification(order)
                
                messages.success(request, f'Payment successful! Order ID: {order.order_id}')
                return redirect('orders')
                
            except razorpay.errors.SignatureVerificationError:
                # Payment verification failed
                order = Order.objects.get(razorpay_order_id=order_id)
                order.status = 'cancelled'
                order.save()
                
                Payment.objects.create(
                    order=order,
                    payment_method='razorpay',
                    payment_status='failed',
                    transaction_id=payment_id,
                    amount=order.grand_total
                )
                
                messages.error(request, 'Payment verification failed! Order cancelled.')
                return redirect('cart')
                
        except Exception as e:
            messages.error(request, f'Payment error: {str(e)}')
            return redirect('cart')
    
    return redirect('index')


@csrf_exempt
def razorpay_webhook(request):
    """Handle Razorpay webhooks for automatic confirmation"""
    if request.method == 'POST':
        try:
            # Get webhook secret from environment (set in Razorpay dashboard)
            webhook_secret = os.environ.get('RAZORPAY_WEBHOOK_SECRET', settings.RAZORPAY_KEY_SECRET)
            webhook_signature = request.headers.get('X-Razorpay-Signature')
            webhook_body = request.body
            
            # Verify webhook signature
            razorpay_client.utility.verify_webhook_signature(
                webhook_body.decode('utf-8'),
                webhook_signature,
                webhook_secret
            )
            
            # Process webhook event
            event_data = json.loads(webhook_body)
            event = event_data.get('event')
            
            if event == 'payment.captured':
                # Payment successful
                payment_entity = event_data['payload']['payment']['entity']
                order_id = payment_entity['notes'].get('order_id')
                
                if order_id:
                    order = Order.objects.get(razorpay_order_id=order_id)
                    order.status = 'confirmed'
                    order.save()
            
            elif event == 'payment.failed':
                # Payment failed
                payment_entity = event_data['payload']['payment']['entity']
                order_id = payment_entity['notes'].get('order_id')
                
                if order_id:
                    order = Order.objects.get(razorpay_order_id=order_id)
                    order.status = 'cancelled'
                    order.save()
            
            return JsonResponse({'status': 'ok'})
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'status': 'invalid method'}, status=405)


def send_sms_notification(contact_id, name, email, phone, message):
    """Placeholder for SMS notification (AWS SNS removed)"""
    print(f"📱 SMS notification (simulated): New contact from {name}")
    return False


def send_email_notification(contact_id, name, email, phone, message):
    """Send email notification for contact form submission"""
    try:
        if not settings.EMAIL_NOTIFICATIONS_ENABLED:
            print("📧 Email notifications disabled in settings")
            return False
            
        # Prepare email content
        subject = f'🍰 New Contact Form - {settings.BAKERY_NAME}'
        
        email_message = f"""
New contact form submission received!

📋 CONTACT DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 Name: {name}
📧 Email: {email}
📱 Phone: {phone if phone else 'Not provided'}
🕒 Submitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🆔 Contact ID: {contact_id}

💬 MESSAGE:
{message}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 QUICK ACTIONS:
• Reply to customer: {email}
• Call customer: {phone if phone else 'No phone provided'}
• View in admin: https://thebakestory.shop/admin/

This notification was sent automatically from {settings.BAKERY_NAME} website.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        # Send email
        send_mail(
            subject=subject,
            message=email_message,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[settings.ADMIN_EMAIL],
            fail_silently=False,
        )
        
        print(f"✅ Email notification sent successfully!")
        print(f"📧 Sent to: {settings.ADMIN_EMAIL}")
        print(f"📋 Subject: {subject}")
        
        return True
        
    except Exception as e:
        print(f"❌ Email notification failed: {str(e)}")
        return False


def send_order_notification_email(order):
    """Send detailed order notification email using AWS SES"""
    try:
        if not settings.ORDER_EMAIL_NOTIFICATIONS_ENABLED:
            print("📧 Order email notifications disabled in settings")
            return False
            
        from django.core.mail import EmailMultiAlternatives
        
        # Get payment status safely
        try:
            payment_status = order.payment.payment_status.upper() if hasattr(order, 'payment') else 'PENDING'
        except:
            payment_status = 'PENDING'
        
        # Calculate order items details
        order_items = order.items.all()
        items_html = ""
        items_text = ""
        
        for item in order_items:
            item_total = item.quantity * item.price
            items_html += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 10px;">{item.menu_item.name}</td>
                    <td style="padding: 10px; text-align: center;">₹{item.price}</td>
                    <td style="padding: 10px; text-align: center;">{item.quantity}</td>
                    <td style="padding: 10px; text-align: right;">₹{item_total}</td>
                </tr>
            """
            items_text += f"• {item.menu_item.name} - ₹{item.price} x {item.quantity} = ₹{item_total}\n"
        
        # Email subject
        subject = f'🍰 NEW ORDER #{order.id} - {settings.BAKERY_BUSINESS_NAME}'
        
        # HTML Email template
        html_message = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: #d2691e; color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 20px; border-radius: 0 0 10px 10px; }}
                .order-details {{ background: white; padding: 15px; margin: 15px 0; border-radius: 8px; border: 1px solid #ddd; }}
                .customer-details {{ background: #e8f4f8; padding: 15px; margin: 15px 0; border-radius: 8px; }}
                .items-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .items-table th {{ background: #d2691e; color: white; padding: 12px; text-align: left; }}
                .total-row {{ background: #fff2e6; font-weight: bold; }}
                .status {{ padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; background: #ffc107; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🍰 NEW ORDER RECEIVED!</h1>
                    <h2>Order #{order.id}</h2>
                </div>
                
                <div class="content">
                    <div class="order-details">
                        <h3>📋 Order Information</h3>
                        <p><strong>Order ID:</strong> #{order.id}</p>
                        <p><strong>Order Date:</strong> {order.created_at.strftime('%d %B %Y at %H:%M')}</p>
                        <p><strong>Status:</strong> <span class="status">{order.status.upper()}</span></p>
                        <p><strong>Payment Status:</strong> {payment_status}</p>
                        <p><strong>Razorpay Order ID:</strong> {order.razorpay_order_id or 'N/A'}</p>
                    </div>
                    
                    <div class="customer-details">
                        <h3>👤 Customer Details</h3>
                        <p><strong>Name:</strong> {order.user.first_name} {order.user.last_name}</p>
                        <p><strong>Email:</strong> {order.user.email}</p>
                        <p><strong>Phone:</strong> {order.delivery_phone}</p>
                        <p><strong>Delivery Address:</strong><br>{order.delivery_address}</p>
                        {f'<p><strong>Special Notes:</strong><br>{order.delivery_notes}</p>' if order.delivery_notes else ''}
                    </div>
                    
                    <div class="order-details">
                        <h3>🛒 Order Items</h3>
                        <table class="items-table">
                            <thead>
                                <tr>
                                    <th>Item</th>
                                    <th style="text-align: center;">Price</th>
                                    <th style="text-align: center;">Quantity</th>
                                    <th style="text-align: right;">Total</th>
                                </tr>
                            </thead>
                            <tbody>
                                {items_html}
                                <tr class="total-row">
                                    <td colspan="3" style="padding: 15px; text-align: right;"><strong>Subtotal:</strong></td>
                                    <td style="padding: 15px; text-align: right;"><strong>₹{order.total_amount}</strong></td>
                                </tr>
                                <tr class="total-row">
                                    <td colspan="3" style="padding: 15px; text-align: right;"><strong>Delivery Fee:</strong></td>
                                    <td style="padding: 15px; text-align: right;"><strong>₹{order.delivery_fee}</strong></td>
                                </tr>
                                <tr class="total-row" style="background: #d2691e; color: white;">
                                    <td colspan="3" style="padding: 15px; text-align: right;"><strong>GRAND TOTAL:</strong></td>
                                    <td style="padding: 15px; text-align: right;"><strong>₹{order.grand_total}</strong></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px; padding: 20px; background: #fff; border-radius: 8px;">
                        <h3 style="color: #d2691e;">Next Steps:</h3>
                        <p>1. Confirm the order with the customer</p>
                        <p>2. Prepare the items for delivery/pickup</p>
                        <p>3. Update order status in admin panel</p>
                        <p>4. Send delivery updates to customer</p>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 20px; color: #666; font-size: 12px;">
                    <p>This notification was sent automatically from {settings.BAKERY_BUSINESS_NAME} Order Management System</p>
                    <p>{settings.BAKERY_BUSINESS_ADDRESS} | {settings.BAKERY_BUSINESS_PHONE}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text version
        text_message = f"""
🍰 NEW ORDER RECEIVED - {settings.BAKERY_BUSINESS_NAME}

ORDER DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 Order ID: #{order.id}
📅 Date: {order.created_at.strftime('%d %B %Y at %H:%M')}
🔄 Status: {order.status.upper()}
💳 Payment: {payment_status}
🔑 Razorpay ID: {order.razorpay_order_id or 'N/A'}

👤 CUSTOMER DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {order.user.first_name} {order.user.last_name}
Email: {order.user.email}
Phone: {order.delivery_phone}
Address: {order.delivery_address}
{f'Notes: {order.delivery_notes}' if order.delivery_notes else ''}

🛒 ORDER ITEMS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{items_text}
                                                    
Subtotal: ₹{order.total_amount}
Delivery Fee: ₹{order.delivery_fee}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRAND TOTAL: ₹{order.grand_total}

📝 NEXT STEPS:
1. Confirm order with customer
2. Prepare items for delivery
3. Update order status in admin panel
4. Send delivery updates to customer

📞 Contact Customer: {order.delivery_phone}
📧 Email Customer: {order.user.email}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{settings.BAKERY_BUSINESS_NAME} | {settings.BAKERY_BUSINESS_ADDRESS}
Phone: {settings.BAKERY_BUSINESS_PHONE}
        """
        
        # Send email using standard Django mail
        send_mail(
            subject=subject,
            message=text_message,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[settings.ORDER_NOTIFICATION_EMAIL],
            html_message=html_message,
            fail_silently=False,
        )
        
        print(f"✅ Order notification email sent for Order #{order.id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send order notification email: {str(e)}")
        return False


def send_order_sms_notification(order):
    """Placeholder for Order SMS notification (AWS SNS removed)"""
    print(f"📱 Order SMS notification (simulated) for Order #{order.id}")
    return False


@csrf_exempt
def submit_contact_form(request):
    """Handle contact form submissions by printing to console (AWS removed)"""
    if request.method == 'POST':
        try:
            # Get form data
            data = json.loads(request.body)
            name = data.get('name', '').strip()
            email = data.get('email', '').strip()
            phone = data.get('phone', '').strip()
            message = data.get('message', '').strip()
            
            # Validate required fields
            if not name or not email or not message:
                return JsonResponse({
                    'success': False,
                    'error': 'Name, email, and message are required'
                }, status=400)
            
            # Generate unique contact ID
            contact_id = str(uuid.uuid4())
            
            print(f"✅ Contact form submitted - ID: {contact_id}")
            print(f"   Name: {name}, Email: {email}, Phone: {phone}")
            print(f"   Message: {message}")
            
            return JsonResponse({
                'success': True,
                'message': 'Thank you for your message! We will get back to you soon.',
                'contact_id': contact_id
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid request data'
            }, status=400)
            
        except Exception as e:
            print(f"❌ Unexpected error in contact form: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': 'Unable to save your message. Please try again later.'
            }, status=500)
            return JsonResponse({
                'success': False,
                'error': 'An error occurred. Please try again later.'
            }, status=500)
    
    return JsonResponse({'error': 'Invalid method'}, status=405)
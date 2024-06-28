from django.db import models
from django.contrib.auth.models import AbstractUser
import random
from django.contrib.auth.hashers import make_password

def generate_unique_account_number():
    try:
        while True:
            account_number = random.randint(100_000_000_000_000, 999_999_999_999_999)
            if not CustomUser.objects.filter(account_no=account_number).exists():
                return account_number
    except:
        return 101

class CustomUser(AbstractUser):
    account_no = models.FloatField(default=generate_unique_account_number)
    savings = models.FloatField(default=0.00)

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='customuser_set',
        related_query_name='customuser'
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='customuser_set',
        related_query_name='customuser'
    )

    def __str__(self):
        return f'{self.username}({self.account_no})'
    
    def save(self, *args, **kwargs):
        if self._state.adding:
            self.password = make_password(self.password)
        super().save(*args, **kwargs)

class CategoryExpenditure(models.Model):
    CATEGORY_CHOICES = [
        ('fd', 'Food'),
        ('Et', 'Entertainment'),
        ('Rt', 'Rent'),
        ('Gr', 'Grocery'),
        ('Ot', 'Other')
    ]
    username = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    expenditure = models.FloatField()
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)

    def __str__(self):
        return f'{self.username}({self.expenditure})'
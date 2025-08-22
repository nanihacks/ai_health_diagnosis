from django.db import models
from django.contrib.auth.models import User


class PredictionHistory(models.Model):
    DISEASE_CHOICES = [
        ('diabetes', 'Diabetes'),
        ('heart', 'Heart Disease'),
        ('alzheimer', "Alzheimer's Disease"),
    ]

    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE)
    disease_type = models.CharField(max_length=20, choices=DISEASE_CHOICES)
    input_data = models.JSONField()
    prediction_result = models.JSONField()
    probability_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.disease_type} prediction @ {self.created_at.strftime('%Y-%m-%d %H:%M')}"

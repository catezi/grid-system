# Generated by Django 3.2 on 2023-04-24 05:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('grid', '0002_auto_20230424_0516'),
    ]

    operations = [
        migrations.AlterField(
            model_name='powergrid',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]

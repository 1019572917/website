# Generated by Django 2.2 on 2020-10-28 16:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('login', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Information',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('opt1', models.CharField(max_length=10)),
                ('opt2', models.CharField(max_length=10)),
                ('input_time', models.FloatField(max_length=10)),
            ],
        ),
    ]

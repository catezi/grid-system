# Generated by Django 3.2 on 2023-04-24 04:33

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PowerGrid',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestep', models.TextField()),
                ('vTime', models.TextField()),
                ('gen_p', models.TextField()),
                ('gen_q', models.TextField()),
                ('gen_v', models.TextField()),
                ('target_dispatch', models.TextField()),
                ('actual_dispatch', models.TextField()),
                ('ld_p', models.TextField()),
                ('adjld_p', models.TextField()),
                ('stoenergy_p', models.TextField()),
                ('ld_q', models.TextField()),
                ('ld_v', models.TextField()),
                ('p_or', models.TextField()),
                ('q_or', models.TextField()),
                ('v_or', models.TextField()),
                ('a_or', models.TextField()),
                ('p_ex', models.TextField()),
                ('q_ex', models.TextField()),
                ('v_ex', models.TextField()),
                ('a_ex', models.TextField()),
                ('line_status', models.TextField()),
                ('grid_loss', models.TextField()),
                ('bus_v', models.TextField()),
                ('bus_gen', models.TextField()),
                ('bus_load', models.TextField()),
                ('bus_branch', models.TextField()),
                ('flag', models.TextField()),
                ('unnameindex', models.TextField()),
                ('action_space', models.TextField()),
                ('steps_to_reconnect_line', models.TextField()),
                ('count_soft_overflow_steps', models.TextField()),
                ('rho', models.TextField()),
                ('gen_status', models.TextField()),
                ('steps_to_recover_gen', models.TextField()),
                ('steps_to_close_gen', models.TextField()),
                ('curstep_renewable_gen_p_max', models.TextField()),
                ('nextstep_renewable_gen_p_max', models.TextField()),
            ],
        ),
    ]

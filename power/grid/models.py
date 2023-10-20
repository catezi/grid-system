
# Create your models here.
from django.db import models
class PowerGrid(models.Model):
    id = models.AutoField(primary_key=True)
    timestep = models.TextField(blank=True, null=True)
    vTime = models.TextField(blank=True, null=True)
    gen_p = models.TextField(blank=True, null=True)
    gen_q = models.TextField(blank=True, null=True)
    gen_v = models.TextField(blank=True, null=True)
    target_dispatch = models.TextField(blank=True, null=True)
    actual_dispatch = models.TextField(blank=True, null=True)
    ld_p = models.TextField(blank=True, null=True)
    adjld_p = models.TextField(blank=True, null=True)
    stoenergy_p = models.TextField(blank=True, null=True)
    ld_q = models.TextField(blank=True, null=True)
    ld_v = models.TextField(blank=True, null=True)
    p_or = models.TextField(blank=True, null=True)
    q_or = models.TextField(blank=True, null=True)
    v_or = models.TextField(blank=True, null=True)
    a_or = models.TextField(blank=True, null=True)
    p_ex = models.TextField(blank=True, null=True)
    q_ex = models.TextField(blank=True, null=True)
    v_ex = models.TextField(blank=True, null=True)
    a_ex = models.TextField(blank=True, null=True)
    line_status = models.TextField(blank=True, null=True)
    grid_loss = models.TextField(blank=True, null=True)
    bus_v = models.TextField(blank=True, null=True)
    bus_gen = models.TextField(blank=True, null=True)
    bus_load = models.TextField(blank=True, null=True)
    bus_branch = models.TextField(blank=True, null=True)
    flag = models.TextField(blank=True, null=True)
    unnameindex = models.TextField(blank=True, null=True)
    action_space = models.TextField(blank=True, null=True)
    steps_to_reconnect_line = models.TextField(blank=True, null=True)
    count_soft_overflow_steps = models.TextField(blank=True, null=True)
    rho = models.TextField(blank=True, null=True)
    gen_status = models.TextField(blank=True, null=True)
    steps_to_recover_gen = models.TextField(blank=True, null=True)
    steps_to_close_gen = models.TextField(blank=True, null=True)
    curstep_renewable_gen_p_max = models.TextField(blank=True, null=True)
    nextstep_renewable_gen_p_max = models.TextField(blank=True, null=True)
    reward = models.TextField(blank=True, null=True)

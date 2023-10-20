from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from .models import PowerGrid
import subprocess
import random
import json

# Create your views here.
def start_power(request):
    # 获取speed参数
    speed = request.GET.get('speed', '1')  # 如果未提供speed，则默认为1
    policy=request.GET.get('policy','DDPG')
    # 执行 Bash 脚本文件，并将speed参数传递给它
    result = subprocess.run(['sh', '/root/qiuyue/gird_system/evaluate_grid_system_DDPG_testBalanceLoss_split_notStateSum_restrictThermalOnOff_useHisState_disconnect_len25.sh', '--speed', speed,'--policy',policy], stdout=subprocess.PIPE)
    # 将输出转换为字符串并打印
    print(result.stdout.decode('utf-8'))
    return HttpResponse("Power is on")


def get_file(name):
    #name=request.GET.get('name')
    try:
        # 获取对应 name 的最后一条数据库信息
        file = PowerGrid.objects.latest('id')
        file_name=getattr(file,name,None)
        # 构建响应数据
        response_data = {
            name:file_name
        }
        # 返回 JSON 响应
        # return JsonResponse(response_data)
        return str(response_data)

    except PowerGrid.DoesNotExist:
        # 如果找不到对应的数据库信息，返回错误响应
        error_response = {
            'error': 'File not found.'
        }
        return JsonResponse(error_response, status=404)
# self.timestep = timestep
# self.vTime = grid.vTime
# self.gen_p = rounded_gen_p
# self.gen_q = grid.prod_q[0]
# self.gen_v = grid.prod_v[0]
# self.target_dispatch = grid.target_dispatch[0]
# self.actual_dispatch = grid.actual_dispatch[0]
# self.ld_p = grid.load_p[0]
# self.adjld_p = [grid.load_p[0][i] for i in settings.adjld_ids]
# self.stoenergy_p = [grid.load_p[0][i] for i in settings.stoenergy_ids]
# self.ld_q = grid.load_q[0]
# self.ld_v = grid.load_v[0]
# self.p_or = grid.p_or[0]
# self.q_or = grid.q_or[0]
# self.v_or = grid.v_or[0]
# self.a_or = grid.a_or[0]
# self.p_ex = grid.p_ex[0]
# self.q_ex = grid.q_ex[0]
# self.v_ex = grid.v_ex[0]
# self.a_ex = grid.a_ex[0]
# self.line_status = grid.line_status[0]
# self.grid_loss = grid.grid_loss
# self.bus_v = grid.bus_v
# # 拓扑结构信息
# self.bus_gen = grid.bus_gen
# self.bus_load = grid.bus_load
# self.bus_branch = grid.bus_branch
# # 潮流计算是否收敛
# self.flag = grid.flag
# self.unnameindex = grid.un_nameindex
# self.action_space = action_space  # 合法动作空间
# self.steps_to_reconnect_line = steps_to_reconnect_line  # 线路断开后恢复连接的剩余时间步数
# self.count_soft_overflow_steps = count_soft_overflow_steps  # 线路软过载的已持续时间步数
# self.rho = rho
# self.gen_status = gen_status  # 机组开关机状态（1为开机，0位关机）
# self.steps_to_recover_gen = steps_to_recover_gen  # 机组关机后可以重新开机的时间步（如果机组状态为开机，则值为0）
# self.steps_to_close_gen = steps_to_close_gen  # 机组开机后可以重新关机的时间步（如果机组状态为关机，则值为0）
# self.curstep_renewable_gen_p_max = curstep_renewable_gen_p_max  # 当前时间步新能源机组的最大有功出力
# self.nextstep_renewable_gen_p_max = nextstep_renewable_gen_p_max  # 下一时间步新能源机组的最大有功出力
def get_timestep(request):
    #从request中获取num
    num=request.Get.get('num','1')

    return HttpResponse(timestep)
def get_vTime(request):
    vTime = get_file('vTime')
    return HttpResponse(vTime)
def get_gen_p(request):
    gen_p = get_file('gen_p')
    return HttpResponse(gen_p)
def get_gen_q(request):
    gen_q = get_file('gen_q')
    return HttpResponse(gen_q)
def get_gen_v(request):
    gen_v = get_file('gen_v')
    return HttpResponse(gen_v)
def get_target_dispatch(request):
    target_dispatch = get_file('target_dispatch')
    return HttpResponse(target_dispatch)
def get_actual_dispatch(request):
    actual_dispatch = get_file('actual_dispatch')
    return HttpResponse(actual_dispatch)
def get_ld_p(request):
    ld_p = get_file('ld_p')
    return HttpResponse(ld_p)
def get_adjld_p(request):
    adjld_p = get_file('adjld_p')
    return HttpResponse(adjld_p)
def get_stoenergy_p(request):
    stoenergy_p = get_file('stoenergy_p')
    return HttpResponse(stoenergy_p)
def get_ld_q(request):
    ld_q = get_file('ld_q')
    return HttpResponse(ld_q)
def get_ld_v(request):
    ld_v = get_file('ld_v')
    return HttpResponse(ld_v)
def get_p_or(request):
    p_or = get_file('p_or')
    return HttpResponse(p_or)
def get_q_or(request):
    q_or = get_file('q_or')
    return HttpResponse(q_or)
def get_v_or(request):
    v_or = get_file('v_or')
    return HttpResponse(v_or)
def get_a_or(request):
    a_or = get_file('a_or')
    return HttpResponse(a_or)
def get_p_ex(request):
    p_ex = get_file('p_ex')
    return HttpResponse(p_ex)
def get_q_ex(request):
    q_ex = get_file('q_ex')
    return HttpResponse(q_ex)
def get_v_ex(request):
    v_ex = get_file('v_ex')
    return HttpResponse(v_ex)
def get_a_ex(request):
    a_ex = get_file('a_ex')
    return HttpResponse(a_ex)
def get_line_status(request):
    file = PowerGrid.objects.latest('id')
    line_status = eval(getattr(file,'line_status',None))
    res = 0
    for i in line_status:
        if i != 'True':
            res+=1
    return HttpResponse(str(res))
def get_grid_loss(request):
    grid_loss = get_file('grid_loss')
    return HttpResponse(grid_loss)
def get_bus_v(request):
    bus_v = get_file('bus_v')
    return HttpResponse(bus_v)
def get_bus_gen(request):
    bus_gen = get_file('bus_gen')
    return HttpResponse(bus_gen)
def get_bus_load(request):
    bus_load = get_file('bus_load')
    return HttpResponse(bus_load)
def get_bus_branch(request):
    bus_branch = get_file('bus_branch')
    return HttpResponse(bus_branch)
def get_flag(request):
    flag = get_file('flag')
    return HttpResponse(flag)
def get_unnameindex(request):
    unnameindex = get_file('unnameindex')
    return HttpResponse(unnameindex)
def get_action_space(request):
    action_space = get_file('action_space')
    return HttpResponse(action_space)
def get_steps_to_reconnect_line(request):
    steps_to_reconnect_line = get_file('steps_to_reconnect_line')
    return HttpResponse(steps_to_reconnect_line)
def get_steps_to_recover_gen(request):
    steps_to_recover_gen = get_file('steps_to_recover_gen')
    return HttpResponse(steps_to_recover_gen)
def get_steps_to_close_gen(request):
    steps_to_close_gen = get_file('steps_to_close_gen')
    return HttpResponse(steps_to_close_gen)
def get_curstep_renewable_gen_p_max(request):
    curstep_renewable_gen_p_max = get_file('curstep_renewable_gen_p_max')
    return HttpResponse(curstep_renewable_gen_p_max)
def get_nextstep_renewable_gen_p_max(request):
    nextstep_renewable_gen_p_max = get_file('nextstep_renewable_gen_p_max')
    return HttpResponse(nextstep_renewable_gen_p_max)

def get_flow(request):
    gen_p = get_file('gen_p').split("[")[1].split("]")[0].split(",")
    flow = [0,0,0,0,0]
    for i in range(len(gen_p)):
        if i%5==0:
            flow[0]+=float(gen_p[i])
        elif i%5==1:
            flow[1]+=float(gen_p[i])
        elif i%5==2:
            flow[2]+=float(gen_p[i])
        elif i%5==3:
            flow[3]+=float(gen_p[i])
        elif i%5==4:
            flow[4]+=float(gen_p[i])
    return HttpResponse(str(flow))

def get_generate_classify(request):

    # renewable1 = [0, 2, 4, 6, 8, 10, 12, 14, 53]
    # renewable2 = [1, 5, 7, 9, 11, 13, 19, 29, 43]
    # thermal_ids = [3, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33,
    # 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52]
    # gen_p = get_file('gen_p').split("[")[1].split("]")[0].split(",")
    flow = [20,40,40]
    # for i in range(len(gen_p)):
    #     if i in renewable1:
    #         flow[0]+=float(gen_p[i])
    #     elif i in renewable2:
    #         flow[1]+=float(gen_p[i])
    #     else:
    #         flow[2]+=float(gen_p[i])
    return HttpResponse(str(flow))

def get_recent_flow(request):
    item_count=min(10,PowerGrid.objects.all().count())
    file_list = PowerGrid.objects.order_by("id").reverse()[:item_count]
    res={}
    for file in file_list:
        file_name = getattr(file,'gen_p',None)
        file_vTime = getattr(file,'vTime',None).split("'")[1].split("'")[0]
        file_id = getattr(file,'id',None)
        print(file_vTime)
        response_data = str({ 'gen_p':file_name })
        gen_p = get_file('gen_p').split("[")[1].split("]")[0].split(",")
        flow = [0,0,0,0,0]
        for i in range(len(gen_p)):
            if i%5==0:
                flow[0]+=float(gen_p[i])
            elif i%5==1:
                flow[1]+=float(gen_p[i])
            elif i%5==2:
                flow[2]+=float(gen_p[i])
            elif i%5==3:
                flow[3]+=float(gen_p[i])
            elif i%5==4:
                flow[4]+=float(gen_p[i])
        res[file_vTime+'_'+str(file_id)] = flow
    return HttpResponse(str(res))

def get_reward(request):
    tmp = PowerGrid.objects.filter()
    reward = 0.0
    for i in tmp:
        reward += float(i.reward)
    return HttpResponse(reward)

def delete_data(request):
    PowerGrid.objects.filter().delete()
    return HttpResponse("已删除上次数据")

def get_bus(request):
    cnt = int(request.GET.get("cnt"))
    file = PowerGrid.objects.order_by("id").reverse()[cnt-1]
    bus_gen = getattr(file,'bus_gen',None)
    bus_load = getattr(file,'bus_load',None)
    bus_branch = getattr(file,'bus_branch',None)
    res = { "bus_gen":eval(bus_gen),
            "bus_load":eval(bus_load),
            "bus_branch":eval(bus_branch)}
    return HttpResponse(JsonResponse(res))

def get_q(request):
    tmp = random.randint(0,100)
    if tmp<3:
        res=2
    elif tmp<10:
        res=1
    else:
        res=0
    return HttpResponse(str(res))

def get_p(request):
    file = PowerGrid.objects.order_by("id").reverse()[0]
    gen_p = eval(getattr(file,"gen_p",None))
    res = gen_p[17]
    return HttpResponse(res)

def get_expense(request):
    tmp = PowerGrid.objects.filter()
    expense = 0.0
    for i in tmp:
        expense += float(i.grid_loss[1:-1])
    return HttpResponse(expense)

def get_new_energy(request):
    renewable_ids = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 29, 43, 53]
    file = PowerGrid.objects.order_by("id").reverse()[0]
    gen_p = eval(getattr(file,"gen_p",None))
    res = {}
    sum = 0
    for i in renewable_ids:
        res[i] = gen_p[i]
        sum += gen_p[i]
    return HttpResponse(sum)

def get_volt(request):
    tmp = random.randint(0,100)
    if tmp<3:
        res=2
    elif tmp<10:
        res=1
    else:
        res=0
    return HttpResponse(str(res))

def get_exception(request):
    res = {"线路":[
                "9:20:09 线路软过载",
                "9:23:09 线路软过载"
            ],
            "平衡机":[
                "9:56:14 平衡机接近极限"
            ],
            "发电机":[
                "9:56:14 发电机达到上限"
            ],
}

    return HttpResponse(JsonResponse(res))


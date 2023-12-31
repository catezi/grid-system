from django.contrib import admin
from django.urls import path
from grid import views
urlpatterns = [
    path('start/',views.start_power),
    path('timestep/', views.get_timestep),
    path('vTime/', views.get_vTime),
    path('gen_p/', views.get_gen_p),
    path('gen_q/', views.get_gen_q),
    path('gen_v/', views.get_gen_v),
    path('target_dispatch/', views.get_target_dispatch),
    path('actual_dispatch/', views.get_actual_dispatch),
    path('ld_p/', views.get_ld_p),
    path('adjld_p/', views.get_adjld_p),
    path('stoenergy_p/', views.get_stoenergy_p),
    path('ld_q/', views.get_ld_q),
    path('ld_v/', views.get_ld_v),
    path('p_or/', views.get_p_or),
    path('q_or/', views.get_q_or),
    path('v_or/', views.get_v_or),
    path('a_or/', views.get_a_or),
    path('p_ex/', views.get_p_ex),
    path('q_ex/', views.get_q_ex),
    path('v_ex/', views.get_v_ex),
    path('a_ex/', views.get_a_ex),
    path('over', views.get_line_status),
    path('grid_loss/', views.get_grid_loss),
    path('bus_v/', views.get_bus_v),
    path('bus_gen/', views.get_bus_gen),
    path('bus_load/', views.get_bus_load),
    path('bus_branch/', views.get_bus_branch),
    path('flag/', views.get_flag),
    path('unnameindex/', views.get_unnameindex),
    path('action_space/', views.get_action_space),
    path('steps_to_reconnect_line/', views.get_steps_to_reconnect_line),
    path('steps_to_recover_gen/', views.get_steps_to_recover_gen),
    path('steps_to_close_gen/', views.get_steps_to_close_gen),
    path('curstep_renewable_gen_p_max/', views.get_curstep_renewable_gen_p_max),
    path('nextstep_renewable_gen_p_max/', views.get_nextstep_renewable_gen_p_max),
    path('flow', views.get_flow),
    path('generate',views.get_generate_classify),
    path('recent_flow', views.get_recent_flow),
    path('reward', views.get_reward),
    path('delete_data', views.delete_data),
    path('history', views.get_bus),
    path('q', views.get_q),
    path('p', views.get_p),
    path('expense', views.get_expense),
    path('new_energy', views.get_new_energy),
    path('volt', views.get_volt),
    path('exception', views.get_exception)
]
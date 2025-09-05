from django.urls import path,include, re_path
from django.contrib import admin
from exam import views
from django.contrib.auth.views import LoginView
from django.conf import settings
from exam.views import admin_upload_questions_pdf_view, admin_upload_subjectwise_view, admin_upload_subjectwise_format2_view
urlpatterns = [
   
    path('admin/', admin.site.urls),
    path('student/',include('student.urls')),
    


    path('',views.home_view,name=''),
    path('logout', views.logout_view, name='logout'),
    path('logout/', views.logout_view),
    path('contactus', views.contactus_view),
    path('afterlogin', views.afterlogin_view,name='afterlogin'),



    path('adminclick', views.adminclick_view),
    path('adminlogin', LoginView.as_view(template_name='exam/adminlogin.html'),name='adminlogin'),
    path('admin-dashboard', views.admin_dashboard_view,name='admin-dashboard'),

    path('admin-student', views.admin_student_view,name='admin-student'),
    path('admin-view-student', views.admin_view_student_view,name='admin-view-student'),
    path('admin-student-statistics', views.admin_student_statistics_view, name='admin-student-statistics'),
    path('admin-student-statistics/<int:student_id>', views.admin_student_statistics_detail_view, name='admin-student-statistics-detail'),
    path('admin-percentile-ranking', views.admin_percentile_ranking_view, name='admin-percentile-ranking'),
    path('delete-batch/<str:batch>', views.delete_batch_view, name='delete-batch'),
    path('admin-view-student-marks', views.admin_view_student_marks_view,name='admin-view-student-marks'),
    path('admin-view-marks/<int:pk>', views.admin_view_marks_view,name='admin-view-marks'),
    path('admin-check-marks/<int:pk>', views.admin_check_marks_view,name='admin-check-marks'),
    path('update-student/<int:pk>', views.update_student_view,name='update-student'),
    path('delete-student/<int:pk>', views.delete_student_view,name='delete-student'),

    path('admin-course', views.admin_course_view,name='admin-course'),
    path('admin-add-course', views.admin_add_course_view,name='admin-add-course'),
    path('admin-view-course', views.admin_view_course_view,name='admin-view-course'),
    path('delete-course/<int:pk>', views.delete_course_view,name='delete-course'),
    path('activate-course/<int:pk>/', views.activate_course_view, name='activate-course'),

    path('admin-question', views.admin_question_view,name='admin-question'),
    path('admin-add-question', views.admin_add_question_view,name='admin-add-question'),
    path('admin-view-question', views.admin_view_question_view,name='admin-view-question'),
    path('view-question/<int:pk>', views.view_question_view,name='view-question'),
    path('delete-question/<int:pk>', views.delete_question_view,name='delete-question'),
    path('upload-questions/', admin_upload_questions_pdf_view, name='admin-upload-questions'),
    path('upload-questions-subjectwise/', admin_upload_subjectwise_view, name='admin-upload-questions-subjectwise'),
    path('upload-questions-subjectwise-format2/', admin_upload_subjectwise_format2_view, name='admin-upload-questions-subjectwise-format2'),
    path('exam/', include('exam.urls')),


]

# Serve MEDIA files in development and optionally in production-like runs under uvicorn/ASGI.
# WhiteNoise handles only STATIC, not MEDIA.
import os
if settings.DEBUG or str(os.environ.get('SERVE_MEDIA', '')).strip().lower() in ('1', 'true', 'yes', 'on'):
    from django.views.static import serve as media_serve
    urlpatterns += [
        re_path(r'^media/(?P<path>.*)$', media_serve, {'document_root': settings.MEDIA_ROOT}),
    ]

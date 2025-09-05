from django.contrib import admin
from django.shortcuts import render, redirect
from django.contrib import messages
from django.urls import path
from django.http import HttpResponseRedirect
from django.core.files.storage import default_storage
from django.conf import settings
import os

from .models import Course, Question, Result, StudentAnswer, QuestionTimeLog
from .pdf_processor import process_exam_pdfs, process_exam_pdfs_format2


class QuestionInline(admin.TabularInline):
    model = Question
    extra = 0
    readonly_fields = ('image',)
    fields = ('question', 'marks', 'correct_answer', 'image')


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ('course_name', 'batch', 'question_number', 'total_marks', 'total_questions', 'duration_minutes', 'status', 'process_pdfs_links')
    list_filter = ('status', 'negative_mark', 'batch')
    search_fields = ('course_name',)
    fields = ('course_name', 'batch', 'question_number', 'total_marks', 'total_questions', 'duration_minutes', 'negative_mark', 'status')
    inlines = [QuestionInline]
    
    def process_pdfs_links(self, obj):
        from django.urls import reverse
        from django.utils.html import format_html
        url_format1 = reverse('admin:exam_course_process_pdfs', args=[obj.pk])
        url_format2 = reverse('admin:exam_course_process_pdfs_format2', args=[obj.pk])
        return format_html(
            '<a class="button" href="{}">Format 1 PDFs</a> '
            '<a class="button" href="{}">Format 2 PDFs</a>', 
            url_format1, url_format2
        )
    process_pdfs_links.short_description = 'PDF Processing'
    process_pdfs_links.allow_tags = True
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                '<int:course_id>/process-pdfs/',
                self.admin_site.admin_view(self.process_pdfs_view),
                name='exam_course_process_pdfs',
            ),
            path(
                '<int:course_id>/process-pdfs-format2/',
                self.admin_site.admin_view(self.process_pdfs_format2_view),
                name='exam_course_process_pdfs_format2',
            ),
        ]
        return custom_urls + urls
    
    def process_pdfs_view(self, request, course_id):
        """Enhanced admin view for processing PDFs with comprehensive reporting"""
        # Debug logging to understand the issue
        print(f"DEBUG: course_id type: {type(course_id)}, value: {course_id}")
        
        # Ensure course_id is an integer
        if not isinstance(course_id, int):
            try:
                course_id = int(course_id)
            except (ValueError, TypeError):
                messages.error(request, f'Invalid course ID: {course_id}')
                return redirect('admin:exam_course_changelist')
        
        try:
            course = Course.objects.get(id=course_id)
            print(f"DEBUG: Retrieved course: {course}, total_questions: {course.total_questions}")
        except Course.DoesNotExist:
            messages.error(request, f'Course with ID {course_id} does not exist')
            return redirect('admin:exam_course_changelist')
        
        if request.method == 'POST':
            questions_pdf = request.FILES.get('questions_pdf')
            answers_pdf = request.FILES.get('answers_pdf')
            clear_existing = request.POST.get('clear_existing') == 'on'
            overwrite_existing = request.POST.get('overwrite_existing') == 'on'
            # Use course.total_questions instead of manual input
            expected_questions = course.total_questions
            print(f"DEBUG: expected_questions type: {type(expected_questions)}, value: {expected_questions}")
            
            # Validate that course has a valid total_questions value
            if not isinstance(expected_questions, int) or expected_questions <= 0:
                messages.error(request, f'Course must have a valid total_questions value (currently: {expected_questions}). Please edit the course and set total_questions > 0.')
                return render(request, 'admin/exam/course/process_pdfs.html', {'course': course})
            
            if not questions_pdf or not answers_pdf:
                messages.error(request, 'Both PDF files are required')
                return render(request, 'admin/exam/course/process_pdfs.html', {'course': course})
            
            try:
                # Save uploaded files temporarily
                questions_path = default_storage.save(
                    f'temp/questions_{course_id}.pdf',
                    questions_pdf
                )
                answers_path = default_storage.save(
                    f'temp/answers_{course_id}.pdf',
                    answers_pdf
                )
                
                # Get full paths
                questions_full_path = os.path.join(settings.MEDIA_ROOT, questions_path)
                answers_full_path = os.path.join(settings.MEDIA_ROOT, answers_path)
                
                # Clear existing questions if requested
                if clear_existing:
                    existing_count = course.question_set.count()
                    course.question_set.all().delete()
                    messages.info(request, f'Deleted {existing_count} existing questions')
                
                # Process PDFs using enhanced OCR-anchor pipeline
                print(f"DEBUG: About to call process_exam_pdfs with:")
                print(f"  questions_full_path: {questions_full_path}")
                print(f"  answers_full_path: {answers_full_path}")
                print(f"  expected_questions: {expected_questions} (type: {type(expected_questions)})")
                print(f"  course: {course} (type: {type(course)})")
                print(f"  overwrite: {overwrite_existing}")
                
                results = process_exam_pdfs(
                    questions_full_path, 
                    answers_full_path, 
                    expected_questions=expected_questions, 
                    course=course,
                    overwrite=overwrite_existing
                )
                
                # Update course question count if successful
                if results.get('success', False):
                    total_questions = results.get('questions_created', 0) + results.get('questions_updated', 0)
                    course.question_number = total_questions
                    course.save()
                
                # Clean up temporary files
                try:
                    default_storage.delete(questions_path)
                    default_storage.delete(answers_path)
                except:
                    pass  # Ignore cleanup errors
                
                # Display results based on pipeline success
                if results.get('success', False):
                    messages.success(
                        request,
                        f'Enhanced processing completed successfully! '
                        f'Created {results.get("questions_created", 0)} questions, '
                        f'updated {results.get("questions_updated", 0)} questions, '
                        f'applied {results.get("answers_updated", 0)} answers.'
                    )
                    
                    # Show pipeline warnings
                    pipeline_report = results.get('pipeline_report', {})
                    for warning in pipeline_report.get('warnings', []):
                        messages.warning(request, f'Warning: {warning}')
                    
                    # Show missing questions if any
                    missing = pipeline_report.get('missing_questions', [])
                    if missing:
                        messages.info(request, f'Missing questions in answers PDF: {", ".join(map(str, missing))}')
                    
                    # Show anchor detection results
                    anchors_found = pipeline_report.get('anchors_found', 0)
                    crops_generated = pipeline_report.get('crops_generated', 0)
                    messages.info(request, f'Detected {anchors_found} question anchors, generated {crops_generated} crops')
                    
                    # Provide link to staging folder for manual review with debug images
                    staging_dir = pipeline_report.get('staging_directory')
                    if staging_dir and os.path.exists(staging_dir):
                        report_path = pipeline_report.get('report_path')
                        if report_path and os.path.exists(report_path):
                            relative_staging = os.path.relpath(staging_dir, settings.MEDIA_ROOT)
                            messages.info(
                                request, 
                                f'Pipeline report and debug images saved to media/{relative_staging}/ for review'
                            )
                    
                    # Warn if question count doesn't match expectations
                    if crops_generated != expected_questions:
                        messages.warning(
                            request,
                            f'Generated {crops_generated} crops but expected {expected_questions} questions. '
                            f'Please review debug images in staging directory.'
                        )
                    
                else:
                    # Show error message
                    error_msg = results.get('error', 'Unknown error occurred')
                    messages.error(request, f'Enhanced processing failed: {error_msg}')
                    
                    # Show pipeline errors if available
                    pipeline_report = results.get('pipeline_report', {})
                    for error in pipeline_report.get('errors', []):
                        messages.error(request, f'Pipeline error: {error}')
                    
                    # Still provide staging directory for debugging even on failure
                    staging_dir = pipeline_report.get('staging_directory')
                    if staging_dir and os.path.exists(staging_dir):
                        relative_staging = os.path.relpath(staging_dir, settings.MEDIA_ROOT)
                        messages.info(request, f'Debug information saved to media/{relative_staging}/ for troubleshooting')
                
                return redirect('admin:exam_course_change', course_id)
                
            except Exception as e:
                messages.error(request, f'Error processing PDFs: {str(e)}')
        
        return render(request, 'admin/exam/course/process_pdfs.html', {'course': course})
    
    def process_pdfs_format2_view(self, request, course_id):
        """Enhanced admin view for processing Format 2 PDFs with comprehensive reporting"""
        # Debug logging to understand the issue
        print(f"DEBUG Format 2: course_id type: {type(course_id)}, value: {course_id}")
        
        # Ensure course_id is an integer
        if not isinstance(course_id, int):
            try:
                course_id = int(course_id)
            except (ValueError, TypeError):
                messages.error(request, f'Invalid course ID: {course_id}')
                return redirect('admin:exam_course_changelist')
        
        try:
            course = Course.objects.get(id=course_id)
            print(f"DEBUG Format 2: Retrieved course: {course}, total_questions: {course.total_questions}")
        except Course.DoesNotExist:
            messages.error(request, f'Course with ID {course_id} does not exist')
            return redirect('admin:exam_course_changelist')
        
        if request.method == 'POST':
            questions_pdf = request.FILES.get('questions_pdf')
            answers_pdf = request.FILES.get('answers_pdf')
            clear_existing = request.POST.get('clear_existing') == 'on'
            overwrite_existing = request.POST.get('overwrite_existing') == 'on'
            # Use course.total_questions instead of manual input
            expected_questions = course.total_questions
            print(f"DEBUG Format 2: expected_questions type: {type(expected_questions)}, value: {expected_questions}")
            
            # Validate that course has a valid total_questions value
            if not isinstance(expected_questions, int) or expected_questions <= 0:
                messages.error(request, f'Course must have a valid total_questions value (currently: {expected_questions}). Please edit the course and set total_questions > 0.')
                return render(request, 'admin/exam/course/process_pdfs_format2.html', {'course': course})
            
            if not questions_pdf or not answers_pdf:
                messages.error(request, 'Both PDF files are required')
                return render(request, 'admin/exam/course/process_pdfs_format2.html', {'course': course})
            
            try:
                # Save uploaded files temporarily
                questions_path = default_storage.save(
                    f'temp/questions_format2_{course_id}.pdf',
                    questions_pdf
                )
                answers_path = default_storage.save(
                    f'temp/answers_format2_{course_id}.pdf',
                    answers_pdf
                )
                
                # Get full paths
                questions_full_path = os.path.join(settings.MEDIA_ROOT, questions_path)
                answers_full_path = os.path.join(settings.MEDIA_ROOT, answers_path)
                
                # Clear existing questions if requested
                if clear_existing:
                    existing_count = course.question_set.count()
                    course.question_set.all().delete()
                    messages.info(request, f'Deleted {existing_count} existing questions')
                
                # Process PDFs using Format 2 pipeline
                print(f"DEBUG Format 2: About to call process_exam_pdfs_format2 with:")
                print(f"  questions_full_path: {questions_full_path}")
                print(f"  answers_full_path: {answers_full_path}")
                print(f"  expected_questions: {expected_questions} (type: {type(expected_questions)})")
                print(f"  course: {course} (type: {type(course)})")
                print(f"  overwrite: {overwrite_existing}")
                
                results = process_exam_pdfs_format2(
                    questions_full_path, 
                    answers_full_path, 
                    expected_questions=expected_questions, 
                    course=course,
                    overwrite=overwrite_existing
                )
                
                # Update course question count if successful
                if results.get('success', False):
                    total_questions = results.get('questions_created', 0) + results.get('questions_updated', 0)
                    course.question_number = total_questions
                    course.save()
                
                # Clean up temporary files
                try:
                    default_storage.delete(questions_path)
                    default_storage.delete(answers_path)
                except:
                    pass  # Ignore cleanup errors
                
                # Display results based on pipeline success
                if results.get('success', False):
                    messages.success(
                        request,
                        f'Format 2 processing completed successfully! '
                        f'Created {results.get("questions_created", 0)} questions, '
                        f'updated {results.get("questions_updated", 0)} questions, '
                        f'applied {results.get("answers_updated", 0)} answers.'
                    )
                    
                    # Show pipeline warnings
                    pipeline_report = results.get('pipeline_report', {})
                    for warning in pipeline_report.get('warnings', []):
                        messages.warning(request, f'Warning: {warning}')
                    
                    # Show missing questions if any
                    missing = pipeline_report.get('missing_questions', [])
                    if missing:
                        messages.info(request, f'Missing questions in Format 2 answers PDF: {", ".join(map(str, missing))}')
                    
                    # Show anchor detection results
                    anchors_found = pipeline_report.get('anchors_found', 0)
                    crops_generated = pipeline_report.get('crops_generated', 0)
                    messages.info(request, f'Format 2: Detected {anchors_found} question anchors, generated {crops_generated} crops')
                    
                    # Provide link to staging folder for manual review with debug images
                    staging_dir = pipeline_report.get('staging_directory')
                    if staging_dir and os.path.exists(staging_dir):
                        report_path = pipeline_report.get('report_path')
                        if report_path and os.path.exists(report_path):
                            relative_staging = os.path.relpath(staging_dir, settings.MEDIA_ROOT)
                            messages.info(
                                request, 
                                f'Format 2 pipeline report and debug images saved to media/{relative_staging}/ for review'
                            )
                    
                    # Warn if question count doesn't match expectations
                    if crops_generated != expected_questions:
                        messages.warning(
                            request,
                            f'Format 2: Generated {crops_generated} crops but expected {expected_questions} questions. '
                            f'Please review debug images in staging directory.'
                        )
                    
                else:
                    # Show error message
                    error_msg = results.get('error', 'Unknown error occurred')
                    messages.error(request, f'Format 2 processing failed: {error_msg}')
                    
                    # Show pipeline errors if available
                    pipeline_report = results.get('pipeline_report', {})
                    for error in pipeline_report.get('errors', []):
                        messages.error(request, f'Format 2 pipeline error: {error}')
                    
                    # Still provide staging directory for debugging even on failure
                    staging_dir = pipeline_report.get('staging_directory')
                    if staging_dir and os.path.exists(staging_dir):
                        relative_staging = os.path.relpath(staging_dir, settings.MEDIA_ROOT)
                        messages.info(request, f'Format 2 debug information saved to media/{relative_staging}/ for troubleshooting')
                
                return redirect('admin:exam_course_change', course_id)
                
            except Exception as e:
                messages.error(request, f'Error processing Format 2 PDFs: {str(e)}')
        
        return render(request, 'admin/exam/course/process_pdfs_format2.html', {'course': course})


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('course', 'question', 'marks', 'correct_answer', 'has_image')
    list_filter = ('course', 'marks', 'correct_answer')
    search_fields = ('question',)
    readonly_fields = ('image_preview',)
    
    def has_image(self, obj):
        return bool(obj.image)
    has_image.boolean = True
    has_image.short_description = 'Has Image'
    
    def image_preview(self, obj):
        if obj.image:
            return f'<img src="{obj.image.url}" style="max-width: 300px; max-height: 200px;"/>'
        return 'No image'
    image_preview.allow_tags = True
    image_preview.short_description = 'Image Preview'


@admin.register(Result)
class ResultAdmin(admin.ModelAdmin):
    list_display = ('student', 'exam', 'marks', 'date', 'violation')
    list_filter = ('exam', 'violation', 'date')
    search_fields = ('student__user__username', 'exam__course_name')
    readonly_fields = ('date',)


@admin.register(StudentAnswer)
class StudentAnswerAdmin(admin.ModelAdmin):
    list_display = ('student', 'exam', 'question', 'selected_option', 'answered_at')
    list_filter = ('exam', 'selected_option', 'answered_at')
    search_fields = ('student__user__username', 'exam__course_name')
    readonly_fields = ('answered_at',)


@admin.register(QuestionTimeLog)
class QuestionTimeLogAdmin(admin.ModelAdmin):
    list_display = ('student', 'question', 'start_time', 'end_time', 'time_spent')
    list_filter = ('start_time', 'end_time')
    search_fields = ('student__user__username', 'question__question')
    readonly_fields = ('time_spent',)

from django.shortcuts import render,redirect,reverse
from . import forms,models
from django.db.models import Sum
from django.contrib.auth.models import Group
from django.http import HttpResponseRedirect
from django.contrib.auth.decorators import login_required,user_passes_test
from django.conf import settings
from datetime import date, timedelta
from django.db.models import Q
from django.core.mail import send_mail
from student import models as SMODEL
from student import forms as SFORM
from django.contrib.auth.models import User
from django.contrib.auth import logout
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .forms import PDFUploadForm
from .utils import parse_pdf_questions
import os
from .models import StudentAnswer
from django.utils import timezone
from .models import QuestionTimeLog
import shutil
import re
# Removed old import - PDF processing now handled by pdf_processor.py


def home_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')  
    return render(request,'exam/index.html')


def is_student(user):
    return user.groups.filter(name='STUDENT').exists()

# Admin access helper: allow superuser, staff, or members of 'ADMIN' group
def is_admin(user):
    try:
        if getattr(user, 'is_superuser', False) or getattr(user, 'is_staff', False):
            return True
        return user.groups.filter(name='ADMIN').exists()
    except Exception:
        return False

def afterlogin_view(request):
    if is_student(request.user):      
        return redirect('student/student-dashboard')
    else:
        return redirect('admin-dashboard')



def adminclick_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')
    return HttpResponseRedirect('adminlogin')


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_dashboard_view(request):
    dict={
    'total_student':SMODEL.Student.objects.all().count(),
    'total_course':models.Course.objects.all().count(),
    'total_question':models.Question.objects.all().count(),
    }
    return render(request,'exam/admin_dashboard.html',context=dict)


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_student_view(request):
    dict={
    'total_student':SMODEL.Student.objects.all().count(),
    }
    return render(request,'exam/admin_student.html',context=dict)

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_view_student_view(request):
    students= SMODEL.Student.objects.all()
    # Group by batch for rendering in 6 sections
    BATCHES = ['11th CET','11th NEET','11th JEE','12th CET','12th NEET','12th JEE']
    students_by_batch = {b: [] for b in BATCHES}
    for s in students:
        key = getattr(s, 'batch', None)
        if key in students_by_batch:
            students_by_batch[key].append(s)
        else:
            # Unknown/missing batch go to first section to preserve visibility
            students_by_batch[BATCHES[0]].append(s)
    return render(request,'exam/admin_view_student.html',{
        'students': students,
        'students_by_batch': students_by_batch,
        'batches': BATCHES,
    })


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_student_statistics_view(request):
    """Batch-wise listing reused for Statistics landing page."""
    students = SMODEL.Student.objects.all().select_related('user')
    BATCHES = ['11th CET','11th NEET','11th JEE','12th CET','12th NEET','12th JEE']
    students_by_batch = {b: [] for b in BATCHES}
    for s in students:
        key = getattr(s, 'batch', None)
        if key in students_by_batch:
            students_by_batch[key].append(s)
        else:
            students_by_batch[BATCHES[0]].append(s)
    return render(request,'exam/admin_student_statistics.html',{
        'students': students,
        'students_by_batch': students_by_batch,
        'batches': BATCHES,
        'is_statistics': True,
    })


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_student_statistics_detail_view(request, student_id: int):
    """Show per-student statistics with charts, computed live from answers/results."""
    from collections import defaultdict
    try:
        student = SMODEL.Student.objects.select_related('user').get(id=student_id)
    except SMODEL.Student.DoesNotExist:
        return render(request, 'exam/admin_view_student.html', {
            'error': 'Student not found.'
        })

    # Subjects considered
    SUBJECTS = ['Physics', 'Chemistry', 'Maths', 'Biology', 'General']
    # All courses attempted by the student (by existence of Result)
    results = models.Result.objects.filter(student=student).select_related('exam')
    course_ids = [r.exam_id for r in results]
    # All questions and answers for these courses
    questions = models.Question.objects.filter(course_id__in=course_ids)
    answers = models.StudentAnswer.objects.filter(student=student, exam_id__in=course_ids)
    ans_by_qid = {a.question_id: a for a in answers}

    # Aggregate subject-wise totals
    subj_totals = {s: {'obtained': 0, 'possible': 0, 'attempted': 0, 'correct': 0, 'incorrect': 0} for s in SUBJECTS}
    per_course_subject_pct = defaultdict(lambda: {s: None for s in SUBJECTS})
    # Compute per-course subject totals for trend graph
    per_course_keyed = defaultdict(lambda: defaultdict(lambda: {'obt': 0, 'poss': 0, 'correct': 0, 'incorrect': 0}))

    # Determine per-course per-question marks
    course_per_q = {}
    for r in results:
        per_q = getattr(r.exam, 'marks_per_question', None) or 1
        course_per_q[r.exam_id] = per_q

    for q in questions:
        subj = q.subject if q.subject in SUBJECTS else 'General'
        per_q = course_per_q.get(q.course_id, 1)
        a = ans_by_qid.get(q.id)
        if a and a.selected_option:
            subj_totals[subj]['attempted'] += 1
            if a.selected_option == q.correct_answer:
                subj_totals[subj]['correct'] += 1
                subj_totals[subj]['obtained'] += per_q
            else:
                subj_totals[subj]['incorrect'] += 1
        # Possible increases regardless of attempt (question exists in that subject)
        subj_totals[subj]['possible'] += per_q
        # Per-course bucket
        bucket = per_course_keyed[q.course_id][subj]
        bucket['poss'] += per_q
        if a and a.selected_option:
            if a.selected_option == q.correct_answer:
                bucket['obt'] += per_q
                bucket['correct'] += 1
            else:
                bucket['incorrect'] += 1

    # Build trend series by course date (or course id if no date)
    # We will sort results by their auto-updated date field
    ordered_results = sorted(results, key=lambda r: (getattr(r, 'date', None) or r.id))
    trend_labels = []
    trend_by_subject = {s: [] for s in SUBJECTS}
    for r in ordered_results:
        label = f"{r.exam.course_name}"
        trend_labels.append(label)
        subj_map = per_course_keyed.get(r.exam_id, {})
        for s in SUBJECTS:
            data = subj_map.get(s, {'obt': 0, 'poss': 0})
            pct = round((data['obt'] / data['poss'] * 100.0), 2) if data['poss'] > 0 else None
            trend_by_subject[s].append(pct)
            per_course_subject_pct[r.exam_id][s] = pct

    # Compute batch averages per subject for comparison
    batch_students = SMODEL.Student.objects.filter(batch=student.batch)
    batch_results = models.Result.objects.filter(student__in=batch_students)
    batch_course_ids = batch_results.values_list('exam_id', flat=True)
    batch_questions = models.Question.objects.filter(course_id__in=batch_course_ids)
    # map course -> per_q
    batch_courses = models.Course.objects.filter(id__in=batch_course_ids)
    b_per_q = {c.id: (getattr(c, 'marks_per_question', None) or 1) for c in batch_courses}
    # student count per subject with data and aggregate totals
    batch_subj_totals = {s: {'obt': 0, 'poss': 0} for s in SUBJECTS}
    batch_answers = models.StudentAnswer.objects.filter(student__in=batch_students, exam_id__in=batch_course_ids)
    b_ans_by_q = {}
    for a in batch_answers:
        b_ans_by_q.setdefault(a.question_id, []).append(a)
    for q in batch_questions:
        subj = q.subject if q.subject in SUBJECTS else 'General'
        per_q = b_per_q.get(q.course_id, 1)
        # Each answer contributes to obtained if correct
        for a in b_ans_by_q.get(q.id, []):
            if a.selected_option:
                if a.selected_option == q.correct_answer:
                    batch_subj_totals[subj]['obt'] += per_q
                # possible for batch increases by per_q per student who attempted/answered
                batch_subj_totals[subj]['poss'] += per_q

    # Prepare data for charts
    subjects_for_ui = [s for s in SUBJECTS if s != 'General'] or SUBJECTS
    subj_obtained = [subj_totals[s]['obtained'] for s in subjects_for_ui]
    subj_possible = [subj_totals[s]['possible'] for s in subjects_for_ui]
    subj_pct = [round((o/p*100.0),2) if p>0 else None for o,p in zip(subj_obtained, subj_possible)]
    batch_pct = []
    for s in subjects_for_ui:
        obt = batch_subj_totals[s]['obt']
        poss = batch_subj_totals[s]['poss']
        batch_pct.append(round((obt/poss*100.0),2) if poss>0 else None)

    # Strength/Weakness: top 2 and bottom 2 by percentage
    pairs = [(s, pct if pct is not None else -1) for s, pct in zip(subjects_for_ui, subj_pct)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    strengths = [p[0] for p in pairs if p[1] >= 0][:2]
    weaknesses = [p[0] for p in reversed(pairs) if p[1] >= 0][:2]

    context = {
        'student': student,
        'subjects': subjects_for_ui,
        'subj_pct': subj_pct,
        'batch_pct': batch_pct,
        'trend_labels': trend_labels,
        'trend_by_subject': trend_by_subject,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'subj_totals': subj_totals,
    }
    return render(request, 'exam/admin_student_statistics_detail.html', context)



@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def update_student_view(request,pk):
    student=SMODEL.Student.objects.get(id=pk)
    user=SMODEL.User.objects.get(id=student.user_id)
    # Use update form: optional password, other fields mapped to User
    userForm=SFORM.StudentUserUpdateForm(instance=user)
    # GET should not pass FILES; instance pre-fills address/contact
    studentForm=SFORM.StudentForm(instance=student)
    mydict={'userForm':userForm,'studentForm':studentForm}
    if request.method=='POST':
        userForm=SFORM.StudentUserUpdateForm(request.POST,instance=user)
        studentForm=SFORM.StudentForm(request.POST,request.FILES,instance=student)
        if userForm.is_valid() and studentForm.is_valid():
            # Save basic user fields first (exclude password from model write)
            user = userForm.save(commit=False)
            # Only set a new password if provided
            new_pwd = userForm.cleaned_data.get('password')
            if new_pwd:
                user.set_password(new_pwd)
            user.save()
            studentForm.save()
            return redirect('admin-view-student')
    return render(request,'exam/update_student.html',context=mydict)



@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def delete_student_view(request,pk):
    student=SMODEL.Student.objects.get(id=pk)
    user=User.objects.get(id=student.user_id)
    user.delete()
    student.delete()
    return HttpResponseRedirect('/admin-view-student')

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def delete_batch_view(request, batch):
    # Only accept POST for destructive action
    if request.method != 'POST':
        return redirect('admin-view-student')
    BATCHES = {'11th CET','11th NEET','11th JEE','12th CET','12th NEET','12th JEE'}
    if batch not in BATCHES:
        return redirect('admin-view-student')
    # Delete all users for students in this batch; cascades remove Student and related data
    qs = SMODEL.Student.objects.filter(batch=batch).select_related('user')
    for s in qs:
        try:
            if s.user_id:
                s.user.delete()
            else:
                s.delete()
        except Exception:
            # continue best-effort
            pass
    return redirect('admin-view-student')


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_course_view(request):
    return render(request,'exam/admin_course.html')


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_add_course_view(request):
    courseForm=forms.CourseForm()
    if request.method=='POST':
        courseForm=forms.CourseForm(request.POST)
        if courseForm.is_valid():        
            course = courseForm.save(commit=False)
            course.status = 'draft'
            course.save()
        else:
            print("form is invalid")
        return HttpResponseRedirect('/admin-view-course')
    return render(request,'exam/admin_add_course.html',{'courseForm':courseForm})


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_view_course_view(request):
    courses = models.Course.objects.all().order_by('-id')  # Latest first
    return render(request,'exam/admin_view_course.html',{'courses':courses})

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def delete_course_view(request,pk):
    course=models.Course.objects.get(id=pk)
    # Local sanitize mirroring models.question_image_upload_path
    def _sanitize(value: str) -> str:
        value = re.sub(r'[^a-zA-Z0-9_]', '_', value or '')
        value = re.sub(r'_+', '_', value)
        value = value.strip('_') or 'unknown'
        return value
    # Delete associated question image files from storage before cascading delete
    try:
        from django.core.files.storage import default_storage
        qs = models.Question.objects.filter(course=course)
        # Collect possible directories to clean up later
        questions_root = os.path.join(settings.MEDIA_ROOT, 'questions')
        legacy_root = os.path.join(settings.MEDIA_ROOT, 'question_images')
        dirs_to_try = set()
        for q in qs:
            if getattr(q, 'image', None) and q.image:
                try:
                    # Record parent directories before deleting file
                    try:
                        file_path = q.image.path
                        # Walk up until reaching roots
                        cur_dir = os.path.dirname(file_path)
                        while cur_dir and os.path.commonpath([cur_dir, questions_root]) == questions_root and cur_dir != questions_root:
                            dirs_to_try.add(cur_dir)
                            cur_dir = os.path.dirname(cur_dir)
                        cur_dir = os.path.dirname(file_path)
                        while cur_dir and os.path.commonpath([cur_dir, legacy_root]) == legacy_root and cur_dir != legacy_root:
                            dirs_to_try.add(cur_dir)
                            cur_dir = os.path.dirname(cur_dir)
                    except Exception:
                        pass
                    # Use the field's delete to remove the file from storage
                    q.image.delete(save=False)
                except Exception:
                    pass
    except Exception:
        # If storage cleanup fails, proceed with DB deletion regardless
        pass
    course.delete()
    # Attempt to remove empty directories collected above (deepest first)
    try:
        for d in sorted(dirs_to_try, key=len, reverse=True):
            try:
                # Remove dir if empty; ignore if not empty
                os.rmdir(d)
            except OSError:
                pass
        # Best-effort: also try course-named folder directly under roots if empty
        candidates = set()
        sanitized = _sanitize(str(course.course_name))
        for root in (os.path.join(settings.MEDIA_ROOT, 'questions'), os.path.join(settings.MEDIA_ROOT, 'question_images')):
            candidates.add(os.path.join(root, str(course.course_name)))
            candidates.add(os.path.join(root, sanitized))
            candidates.add(os.path.join(root, str(course.id)))
        for cand in sorted(candidates, key=len, reverse=True):
            try:
                os.rmdir(cand)
            except OSError:
                pass
    except Exception:
        pass
    return HttpResponseRedirect('/admin-view-course')



@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_question_view(request):
    return render(request,'exam/admin_question.html')


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_add_question_view(request):
    questionForm=forms.QuestionForm()
    if request.method=='POST':
        questionForm=forms.QuestionForm(request.POST, request.FILES)
        if questionForm.is_valid():
            question=questionForm.save(commit=False)
            course=models.Course.objects.get(id=request.POST.get('courseID'))
            question.course=course
            question.save()
            # Update course question_number and total_marks using course marks_per_question
            questions = models.Question.objects.filter(course=course)
            course.question_number = questions.count()
            per_q = getattr(course, 'marks_per_question', None) or 1
            course.total_marks = course.question_number * per_q
            course.save()
            if 'save_next' in request.POST:
                return redirect('admin-add-question')
            # Default: go to list of questions for this course
            return redirect('view-question', pk=course.id)
        else:
            # Stay on the same page and display errors
            return render(request,'exam/admin_add_question.html',{'questionForm':questionForm})
        # Fallback
        return HttpResponseRedirect('/admin-view-question')
    return render(request,'exam/admin_add_question.html',{'questionForm':questionForm})


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_view_question_view(request):
    courses= models.Course.objects.all().order_by('-id')  # Latest first
    return render(request,'exam/admin_view_question.html',{'courses':courses})

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def view_question_view(request,pk):
    questions=models.Question.objects.all().filter(course_id=pk)
    return render(request,'exam/view_question.html',{'questions':questions})

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def delete_question_view(request,pk):
    question=models.Question.objects.get(id=pk)
    course = question.course
    question.delete()
    # Update course question_number and total_marks
    questions = models.Question.objects.filter(course=course)
    course.question_number = questions.count()
    per_q = getattr(course, 'marks_per_question', None) or 1
    course.total_marks = course.question_number * per_q
    course.save()
    # Stay on the same course question list page after deletion
    return redirect('view-question', pk=course.id)

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_view_student_marks_view(request):
    students= SMODEL.Student.objects.all()
    return render(request,'exam/admin_view_student_marks.html',{'students':students})


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_percentile_ranking_view(request):
    # Build rankings per test (course) dynamically from Results
    # Fetch all results with related objects to minimize queries
    results = models.Result.objects.select_related('student__user', 'exam')
    # Group results by course
    from collections import defaultdict
    by_course = defaultdict(list)
    for r in results:
        by_course[r.exam].append(r)

    rankings_by_course = {}
    ordered_courses = []
    for course, res_list in by_course.items():
        if not res_list:
            continue
        # Sort by marks desc
        res_list.sort(key=lambda x: x.marks, reverse=True)
        total = len(res_list)
        # Precompute counts for percentile with ties handled (<= score)
        # Build frequency map of marks
        from collections import Counter
        freq = Counter(r.marks for r in res_list)
        # Build cumulative counts for <= score
        sorted_marks = sorted(freq.keys())
        cumulative = {}
        running = 0
        for m in sorted_marks:
            running += freq[m]
            cumulative[m] = running
        # Assign rank (dense rank: increments when marks drop)
        rows = []
        last_marks = None
        current_rank = 0
        for r in res_list:
            if last_marks is None or r.marks != last_marks:
                current_rank += 1
                last_marks = r.marks
            percentile = round((cumulative[r.marks] / total) * 100, 2)
            # Ensure top rank shows 100.00 explicitly
            if current_rank == 1:
                percentile = 100.00
            rows.append({
                'rank': current_rank,
                'student_name': r.student.get_name,
                'marks': r.marks,
                'percentile': percentile,
            })
        rankings_by_course[course.id] = rows
        ordered_courses.append(course)

    # Sort courses by name for stable UI
    ordered_courses.sort(key=lambda c: c.course_name)
    return render(request, 'exam/admin_view_student_marks.html', {
        'rankings_by_course': rankings_by_course,
        'ordered_courses': ordered_courses,
    })

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_view_marks_view(request,pk):
    courses = models.Course.objects.all().order_by('-id')  # Latest first
    student = SMODEL.Student.objects.get(id=pk)
    
    # Get marks for each course for this student
    courses_with_marks = []
    for course in courses:
        try:
            result = models.Result.objects.get(student=student, exam=course)
            marks_obtained = result.marks
        except models.Result.DoesNotExist:
            marks_obtained = "Not Attempted"
        
        courses_with_marks.append({
            'course': course,
            'marks_obtained': marks_obtained
        })
    
    response = render(request,'exam/admin_view_marks.html',{
        'courses_with_marks': courses_with_marks,
        'student': student
    })
    response.set_cookie('student_id',str(pk))
    return response

@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_check_marks_view(request,pk):
    course = models.Course.objects.get(id=pk)
    student_id = request.COOKIES.get('student_id')
    if not student_id:
        return render(request, 'exam/admin_check_marks.html', {
            'error': 'No student selected. Please use the admin interface to select a student.'
        })
    try:
        student = SMODEL.Student.objects.get(id=student_id)
    except SMODEL.Student.DoesNotExist:
        return render(request, 'exam/admin_check_marks.html', {
            'error': 'Student not found. Please check your selection.'
        })
    results= models.Result.objects.all().filter(exam=course).filter(student=student)
    # Full question set for this exam
    questions = models.Question.objects.filter(course=course).select_related('course').order_by('question_number', 'id')
    # Total possible marks similar to student view, based on course marks_per_question
    per_q = getattr(course, 'marks_per_question', None) or 1
    total_possible_marks = len(questions) * per_q
    answers = StudentAnswer.objects.filter(student=student, exam=course).select_related('question')
    answers_by_qid = {a.question_id: a for a in answers}
    # New: Build a mapping of question.id -> time_taken (in seconds)
    qtl_by_qid = {}
    for q in questions:
        try:
            qtl = QuestionTimeLog.objects.get(student=student, question=q)
            if qtl.time_spent:
                qtl_by_qid[q.id] = int(qtl.time_spent.total_seconds())
            else:
                qtl_by_qid[q.id] = "-"
        except QuestionTimeLog.DoesNotExist:
            qtl_by_qid[q.id] = "-"
    # Subject-wise breakdown similar to student check-marks
    SUBJECT_ORDER = ['Physics', 'Chemistry', 'Maths', 'Biology']
    present_subjects = list(models.Question.objects.filter(course=course).values_list('subject', flat=True).distinct())
    non_general = [s for s in present_subjects if s and s != 'General']
    ordered_subjects = [s for s in SUBJECT_ORDER if s in non_general]
    subject_stats = []
    if ordered_subjects:
        for subj in ordered_subjects:
            subj_questions = [q for q in questions if q.subject == subj]
            correct = 0
            incorrect = 0
            subj_marks = 0
            for q in subj_questions:
                ans = answers_by_qid.get(q.id)
                selected_letter = ans.selected_option if ans else None
                if selected_letter == q.correct_answer:
                    correct += 1
                    subj_marks += per_q
                elif selected_letter is not None and selected_letter != q.correct_answer:
                    incorrect += 1
            if hasattr(course, 'negative_mark') and course.negative_mark:
                subj_marks = max(0, subj_marks - incorrect * per_q)
            subject_stats.append({
                'subject': subj,
                'total': len(subj_questions),
                'correct': correct,
                'incorrect': incorrect,
                'marks': subj_marks,
            })
    # Build questions grouped by subject for sectioned display like student view
    questions_by_subject = {}
    for subj in (ordered_subjects or []):
        qs = [q for q in questions if q.subject == subj]
        qs.sort(key=lambda q: (q.question_number or 0, q.id))
        questions_by_subject[subj] = qs

    return render(request,'exam/admin_check_marks.html',{
        'results': results,
        'questions': questions,
        'all_questions': questions,
        'questions_by_subject': questions_by_subject,
        'total_possible_marks': total_possible_marks,
        'answers_by_qid': answers_by_qid,
        'student': student,
        'course': course,
        'qtl_by_qid': qtl_by_qid,
        'subject_stats': subject_stats,
    })
    




def aboutus_view(request):
    return render(request,'exam/aboutus.html')

def logout_view(request):
    """Log out the user on GET or POST and redirect to home."""
    try:
        logout(request)
    except Exception:
        # Ignore any logout edge-case errors and still redirect
        pass
    return redirect('/')

def contactus_view(request):
    sub = forms.ContactusForm()
    if request.method == 'POST':
        sub = forms.ContactusForm(request.POST)
        if sub.is_valid():
            email = sub.cleaned_data['Email']
            name=sub.cleaned_data['Name']
            message = sub.cleaned_data['Message']
            send_mail(str(name)+' || '+str(email),message,settings.EMAIL_HOST_USER, settings.EMAIL_RECEIVING_USER, fail_silently = False)
            return render(request, 'exam/contactussuccess.html')
    return render(request, 'exam/contactus.html', {'form':sub})


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def activate_course_view(request, pk):
    if request.method == 'POST':
        course = models.Course.objects.get(id=pk)
        course.status = 'active'
        # Set global activation timestamp so duration starts for everyone now
        if not getattr(course, 'activated_at', None):
            course.activated_at = timezone.now()
        # Update question_number and total_marks
        questions = models.Question.objects.filter(course=course)
        course.question_number = questions.count()
        per_q = getattr(course, 'marks_per_question', None) or 1
        course.total_marks = course.question_number * per_q
        course.save()
    return redirect('admin-view-course')


@login_required(login_url='adminlogin')
@user_passes_test(is_admin)
def admin_upload_questions_pdf_view(request):
    """Two-step PDF processing: Questions PDF first, then Answers PDF"""
    from .pdf_processor import process_exam_pdfs
    from django.contrib import messages
    from django.core.files.storage import default_storage
    
    courses = models.Course.objects.all().order_by('-id')  # Latest first
    
    # Clear any existing messages on GET request (fresh page load)
    if request.method == 'GET':
        # Clear all messages from previous sessions
        storage = messages.get_messages(request)
        for _ in storage:
            pass  # Consume all messages to clear them
    
    if request.method == 'POST':
        course_id = request.POST.get('course')
        questions_pdf = request.FILES.get('questions_pdf')
        answers_pdf = request.FILES.get('answers_pdf')
        clear_existing = request.POST.get('clear_existing') == 'on'
        
        if not course_id:
            messages.error(request, 'Please select a course')
            return render(request, 'exam/upload_pdf_two_step.html', {'courses': courses})
            
        if not questions_pdf or not answers_pdf:
            messages.error(request, 'Both Questions PDF and Answers PDF are required')
            return render(request, 'exam/upload_pdf_two_step.html', {'courses': courses})
        
        try:
            course = models.Course.objects.get(id=course_id)

            # Save uploaded files temporarily (default_storage may be FileSystemStorage)
            questions_path = default_storage.save(f'temp/questions_{course_id}.pdf', questions_pdf)
            answers_path = default_storage.save(f'temp/answers_{course_id}.pdf', answers_pdf)

            # Robustly obtain absolute filesystem path (works with FileSystemStorage)
            def get_full_path(storage, rel_path):
                if hasattr(storage, 'path'):
                    try:
                        return storage.path(rel_path)
                    except Exception:
                        pass
                # Fallback to MEDIA_ROOT join (works for default local storage)
                return os.path.join(settings.MEDIA_ROOT, rel_path)

            questions_full_path = get_full_path(default_storage, questions_path)
            answers_full_path = get_full_path(default_storage, answers_path)

            # Clear existing questions if requested
            if clear_existing:
                existing_count = course.question_set.count()
                course.question_set.all().delete()
                messages.info(request, f'Deleted {existing_count} existing questions')

            # Optional overwrite flag (if you add checkbox to the form)
            overwrite_existing = request.POST.get('overwrite_existing') == 'on'

            # Optional manual expected_questions override from form (recommended)
            expected_override = request.POST.get('expected_questions')
            try:
                expected_override = int(expected_override) if expected_override not in (None, '') else None
            except ValueError:
                expected_override = None

            # Derive expected_questions in a safe order: form override -> course.total_questions -> course.question_number -> current DB count
            expected_questions = None
            if expected_override:
                expected_questions = expected_override
            else:
                for attr in ('total_questions', 'question_number'):
                    v = getattr(course, attr, None)
                    if isinstance(v, int) and v > 0:
                        expected_questions = v
                        break
                if not expected_questions:
                    cnt = course.question_set.count()
                    if cnt > 0:
                        expected_questions = cnt

            # Validate expected_questions (processor expects a positive integer for completeness checks)
            if not isinstance(expected_questions, int) or expected_questions <= 0:
                # Clean up temp files
                try:
                    default_storage.delete(questions_path)
                    default_storage.delete(answers_path)
                except Exception:
                    pass
                messages.error(
                    request,
                    f'Course must have a valid total_questions value or you must provide an Expected Questions number. '
                    f'Currently: {expected_questions}'
                )
                return render(request, 'exam/upload_pdf_two_step.html', {'courses': courses})

            # Call the processor with named args (prevents wrong positional passing)
            results = process_exam_pdfs(
                questions_full_path,
                answers_full_path,
                expected_questions=expected_questions,
                course=course,
                overwrite=overwrite_existing
            )

            # Normalize results so template doesn't see empty values
            questions_created = results.get('questions_created', results.get('created', 0))
            questions_updated = results.get('questions_updated', results.get('updated', 0))
            answers_updated = results.get('answers_updated', results.get('answers_updated', 0))
            skipped_questions = results.get('skipped_questions', []) or results.get('pipeline_report', {}).get('skipped_questions', [])

            # Update course question_number (created + updated) if pipeline succeeded
            if results.get('success', False):
                course.question_number = questions_created + questions_updated
                course.save()

            # Push friendly messages and pipeline warnings/errors
            if results.get('success', False):
                messages.success(
                    request,
                    f'Processing completed successfully! Created {questions_created} questions, '
                    f'updated {questions_updated} questions, applied {answers_updated} answers.'
                )

                # show pipeline-level warnings
                pipeline_report = results.get('pipeline_report', {}) or {}
                for warning in pipeline_report.get('warnings', []):
                    messages.warning(request, f'Warning: {warning}')

                # show missing / skipped questions to the admin
                missing = pipeline_report.get('missing_questions', []) or []
                if missing:
                    messages.info(request, f'Missing questions in answers PDF: {", ".join(map(str, missing))}')
                if skipped_questions:
                    messages.info(request, f'Skipped questions while cropping: {", ".join(map(str, skipped_questions))}')

            else:
                # failure path: display errors from pipeline_report if present
                messages.error(request, f'Error during processing: {results.get("error", "Unknown error")}')
                pipeline_report = results.get('pipeline_report', {}) or {}
                for err in pipeline_report.get('errors', []):
                    messages.error(request, f'Pipeline error: {err}')

            # Finally render the result page with the full results dict (so template can access everything)
            return render(request, 'exam/upload_result_two_step.html', {
                'results': results,
                'course': course,
            })

        finally:
            # Ensure temp uploads are cleaned up in all cases (best-effort)
            try:
                default_storage.delete(questions_path)
                default_storage.delete(answers_path)
            except Exception:
                pass

            except models.Course.DoesNotExist:
                messages.error(request, f'Course with ID {course_id} does not exist')
            except Exception as e:
                # Catch-all: ensure temporary files removed and message shown
                try:
                    default_storage.delete(questions_path)
                    default_storage.delete(answers_path)
                except Exception:
                    pass
                messages.error(request, f'Error processing PDFs: {str(e)}')
    
    return render(request, 'exam/upload_pdf_two_step.html', {'courses': courses})


@login_required(login_url='adminlogin')
def admin_upload_subjectwise_view(request):
    """Subject-wise upload: 4 subjects x (Questions, Answers). Allows partial subjects with pair validation."""
    from .pdf_processor import process_exam_pdfs
    from django.contrib import messages
    from django.core.files.storage import default_storage

    SUBJECTS = ['Physics', 'Chemistry', 'Maths', 'Biology', 'Mix']
    courses = models.Course.objects.all().order_by('-id')  # Latest first

    if request.method == 'POST':
        course_id = request.POST.get('course')
        if not course_id:
            messages.error(request, 'Please select a course')
            return render(request, 'exam/upload_pdf_subjectwise.html', {'courses': courses, 'subjects': SUBJECTS})

        try:
            course = models.Course.objects.get(id=course_id)
        except models.Course.DoesNotExist:
            messages.error(request, 'Invalid course')
            return render(request, 'exam/upload_pdf_subjectwise.html', {'courses': courses, 'subjects': SUBJECTS})

        # Options
        overwrite_existing = request.POST.get('overwrite_existing') == 'on'
        expected_override = request.POST.get('expected_questions')
        try:
            expected_override = int(expected_override) if expected_override not in (None, '') else None
        except ValueError:
            expected_override = None

        per_subject_results = {}
        any_uploaded = False

        for subj in SUBJECTS:
            q_file = request.FILES.get(f'{subj.lower()}_questions')
            a_file = request.FILES.get(f'{subj.lower()}_answers')

            # Skip subject if no files provided
            if not q_file and not a_file:
                continue

            any_uploaded = True

            # Pair validation
            if (q_file and not a_file) or (a_file and not q_file):
                per_subject_results[subj] = {
                    'success': False,
                    'error': 'Both Questions and Answers PDFs are required for this subject'
                }
                continue

            # Save temp files
            questions_path = default_storage.save(f'temp/{subj}_questions_{course_id}.pdf', q_file)
            answers_path = default_storage.save(f'temp/{subj}_answers_{course_id}.pdf', a_file)

            def get_full_path(storage, rel_path):
                if hasattr(storage, 'path'):
                    try:
                        return storage.path(rel_path)
                    except Exception:
                        pass
                return os.path.join(settings.MEDIA_ROOT, rel_path)

            q_full = get_full_path(default_storage, questions_path)
            a_full = get_full_path(default_storage, answers_path)

            # Derive expected_questions per subject
            expected_questions = expected_override
            if not expected_questions:
                v = getattr(course, 'total_questions', None)
                if isinstance(v, int) and v > 0:
                    expected_questions = v
            if not expected_questions:
                expected_questions = None

            # Run per-subject in its own transaction and try/finally for cleanup
            try:
                result = process_exam_pdfs(
                    q_full,
                    a_full,
                    expected_questions=expected_questions if expected_questions else 1,
                    course=course,
                    overwrite=overwrite_existing,
                    subject=('General' if subj == 'Mix' else subj)
                )
                per_subject_results[subj] = result
            except Exception as e:
                per_subject_results[subj] = {'success': False, 'error': str(e)}
            finally:
                try:
                    default_storage.delete(questions_path)
                    default_storage.delete(answers_path)
                except Exception:
                    pass

        if not any_uploaded:
            messages.error(request, 'Please upload at least one subject pair or use the legacy mixed-PDF mode.')
            return render(request, 'exam/upload_pdf_subjectwise.html', {'courses': courses, 'subjects': SUBJECTS})

        # Aggregate messages
        for subj, res in per_subject_results.items():
            if res.get('success'):
                messages.success(request, f"{subj}: Created {res.get('questions_created',0)}, Updated {res.get('questions_updated',0)}, Answers {res.get('answers_updated',0)}")
            else:
                messages.error(request, f"{subj}: {res.get('error','Failed to process')}")

        return render(request, 'exam/upload_result_subjectwise.html', {
            'results_by_subject': per_subject_results,
            'course': course,
        })

    return render(request, 'exam/upload_pdf_subjectwise.html', {'courses': courses, 'subjects': SUBJECTS})


@login_required(login_url='adminlogin')
def admin_upload_subjectwise_format2_view(request):
    """Subject-wise upload UI for Format 2 processing. Identical UI, separate route to avoid mixing formats."""
    from .pdf_processor import process_exam_pdfs_format2
    from django.contrib import messages
    from django.core.files.storage import default_storage

    SUBJECTS = ['Physics', 'Chemistry', 'Maths', 'Biology', 'Mix']
    courses = models.Course.objects.all().order_by('-id')  # Latest first

    if request.method == 'POST':
        course_id = request.POST.get('course')
        if not course_id:
            messages.error(request, 'Please select a course')
            return render(request, 'exam/upload_pdf_subjectwise_format2.html', {'courses': courses, 'subjects': SUBJECTS})

        try:
            course = models.Course.objects.get(id=course_id)
        except models.Course.DoesNotExist:
            messages.error(request, 'Invalid course')
            return render(request, 'exam/upload_pdf_subjectwise_format2.html', {'courses': courses, 'subjects': SUBJECTS})

        overwrite_existing = request.POST.get('overwrite_existing') == 'on'
        expected_override = request.POST.get('expected_questions')
        try:
            expected_override = int(expected_override) if expected_override not in (None, '') else None
        except ValueError:
            expected_override = None

        per_subject_results = {}
        any_uploaded = False

        for subj in SUBJECTS:
            q_file = request.FILES.get(f'{subj.lower()}_questions')
            a_file = request.FILES.get(f'{subj.lower()}_answers')

            if not q_file and not a_file:
                continue

            any_uploaded = True

            if (q_file and not a_file) or (a_file and not q_file):
                per_subject_results[subj] = {
                    'success': False,
                    'error': 'Both Questions and Answers PDFs are required for this subject'
                }
                continue

            questions_path = default_storage.save(f'temp/{subj}_questions_{course_id}.pdf', q_file)
            answers_path = default_storage.save(f'temp/{subj}_answers_{course_id}.pdf', a_file)

            def get_full_path(storage, rel_path):
                if hasattr(storage, 'path'):
                    try:
                        return storage.path(rel_path)
                    except Exception:
                        pass
                return os.path.join(settings.MEDIA_ROOT, rel_path)

            q_full = get_full_path(default_storage, questions_path)
            a_full = get_full_path(default_storage, answers_path)

            expected_questions = expected_override
            if not expected_questions:
                v = getattr(course, 'total_questions', None)
                if isinstance(v, int) and v > 0:
                    expected_questions = v
            if not expected_questions:
                expected_questions = None

            try:
                result = process_exam_pdfs_format2(
                    q_full,
                    a_full,
                    expected_questions=expected_questions if expected_questions else 1,
                    course=course,
                    overwrite=overwrite_existing,
                    subject=('General' if subj == 'Mix' else subj)
                )
                per_subject_results[subj] = result
            except Exception as e:
                per_subject_results[subj] = {'success': False, 'error': str(e)}
            finally:
                try:
                    default_storage.delete(questions_path)
                    default_storage.delete(answers_path)
                except Exception:
                    pass

        if not any_uploaded:
            messages.error(request, 'Please upload at least one subject pair for Format 2.')
            return render(request, 'exam/upload_pdf_subjectwise_format2.html', {'courses': courses, 'subjects': SUBJECTS})

        for subj, res in per_subject_results.items():
            if res.get('success'):
                messages.success(request, f"{subj} (Format 2): Created {res.get('questions_created',0)}, Updated {res.get('questions_updated',0)}, Answers {res.get('answers_updated',0)}")
            else:
                messages.error(request, f"{subj} (Format 2): {res.get('error','Failed to process')}")

        return render(request, 'exam/upload_result_subjectwise.html', {
            'results_by_subject': per_subject_results,
            'course': course,
        })

    return render(request, 'exam/upload_pdf_subjectwise_format2.html', {'courses': courses, 'subjects': SUBJECTS})



@csrf_exempt
def exam_log_view(request):
    if request.method == 'POST':
        import json
        try:
            data = json.loads(request.body.decode())
            print('[ExamTimingLog]', data.get('msg', ''))
        except Exception as e:
            print('[ExamTimingLog] Error:', e)
        return JsonResponse({'status': 'ok'})
    return JsonResponse({'status': 'invalid'}, status=400)


@csrf_exempt
@login_required(login_url='studentlogin')
def question_time_log_view(request):
    if request.method == 'POST':
        try:
            import json
            data = json.loads(request.body.decode())
            question_id = data.get('question_id')
            event_type = data.get('event_type')
            if not question_id or event_type not in ['enter', 'exit']:
                return JsonResponse({'status': 'error', 'msg': 'Invalid data'}, status=400)
            from student.models import Student
            student = Student.objects.get(user_id=request.user.id)
            from .models import Question, QuestionTimeLog
            question = Question.objects.get(id=question_id)
            now = timezone.now()
            if event_type == 'enter':
                # Only create if not exists
                if not QuestionTimeLog.objects.filter(student=student, question=question).exists():
                    QuestionTimeLog.objects.create(student=student, question=question, start_time=now)
            elif event_type == 'exit':
                try:
                    log = QuestionTimeLog.objects.get(student=student, question=question)
                    log.end_time = now
                    log.save()
                except QuestionTimeLog.DoesNotExist:
                    # Fallback: create with both times if somehow missing
                    QuestionTimeLog.objects.create(student=student, question=question, start_time=now, end_time=now)
            return JsonResponse({'status': 'ok'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'msg': str(e)}, status=500)
    return JsonResponse({'status': 'invalid'}, status=400)



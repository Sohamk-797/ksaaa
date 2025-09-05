from django.shortcuts import render,redirect,reverse
from . import forms,models
from django.db.models import Sum, Q
from django.contrib.auth.models import Group
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth.decorators import login_required,user_passes_test
from django.conf import settings
from datetime import date, timedelta
from exam import models as QMODEL
import random
from exam.models import StudentAnswer, QuestionTimeLog  # Import the new models
import json


#for showing signup/login button for student
def studentclick_view(request):
    if request.user.is_authenticated:
        return HttpResponseRedirect('afterlogin')
    return render(request,'student/studentclick.html')

def student_signup_view(request):
    userForm = forms.StudentUserForm()
    studentForm = forms.StudentForm()
    mydict = {'userForm': userForm, 'studentForm': studentForm}
    if request.method == 'POST':
        userForm = forms.StudentUserForm(request.POST)
        studentForm = forms.StudentForm(request.POST, request.FILES)
        if userForm.is_valid() and studentForm.is_valid():
            user = userForm.save()
            user.set_password(user.password)
            user.save()
            student = studentForm.save(commit=False)
            student.user = user
            student.save()
            my_student_group = Group.objects.get_or_create(name='STUDENT')
            my_student_group[0].user_set.add(user)
            return HttpResponseRedirect('studentlogin')
        # On validation errors, fall through and re-render with errors
        mydict = {'userForm': userForm, 'studentForm': studentForm}
        return render(request, 'student/studentsignup.html', context=mydict)
    return render(request, 'student/studentsignup.html', context=mydict)

def is_student(user):
    return user.groups.filter(name='STUDENT').exists()

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_dashboard_view(request):
    dict={
    
    'total_course':QMODEL.Course.objects.all().count(),
    'total_question':QMODEL.Question.objects.all().count(),
    }
    return render(request,'student/student_dashboard.html',context=dict)

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_exam_view(request):
    # Only show active courses for the student's batch
    try:
        me = models.Student.objects.get(user_id=request.user.id)
        courses = QMODEL.Course.objects.filter(status='active', batch=me.batch).order_by('-id')  # Latest first
    except models.Student.DoesNotExist:
        courses = QMODEL.Course.objects.none()
    return render(request,'student/student_exam.html',{'courses':courses})

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def take_exam_view(request,pk):
    course=QMODEL.Course.objects.get(id=pk)
    student = None
    try:
        student = models.Student.objects.get(user_id=request.user.id)
    except models.Student.DoesNotExist:
        pass
    # Enforce batch-level access control
    if student and course.batch != student.batch:
        return HttpResponse("This test is not assigned to your batch.", status=403)
    if student and QMODEL.Result.objects.filter(student=student, exam=course).exists():
        return redirect('check-marks', pk=course.id)
    total_questions=QMODEL.Question.objects.all().filter(course=course).count()
    questions=QMODEL.Question.objects.all().filter(course=course)
    # Prefer course.marks_per_question for total marks to keep consistent scoring
    per_q = getattr(course, 'marks_per_question', None) or 1
    total_marks = total_questions * per_q
    
    response = render(request,'student/take_exam.html',{'course':course,'total_questions':total_questions,'total_marks':total_marks})
    # Prevent caching so back button cannot return to this page after submission
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def start_exam_view(request, pk):
    course = QMODEL.Course.objects.get(id=pk)
    try:
        student = models.Student.objects.get(user_id=request.user.id)
    except models.Student.DoesNotExist:
        return HttpResponse("No student profile found for this user. Please contact admin.", status=403)
    # Enforce batch-level access control
    if course.batch != student.batch:
        return HttpResponse("This test is not assigned to your batch.", status=403)
    # HARD BLOCK: If result exists, never allow access back to start exam
    if QMODEL.Result.objects.filter(student=student, exam=course).exists():
        return redirect('check-marks', pk=course.id)

    # Determine available subjects for this course (stable ordered unique list)
    SUBJECT_ORDER = ['Physics', 'Chemistry', 'Maths', 'Biology']
    present_set = set(
        QMODEL.Question.objects
        .filter(course=course, subject__in=SUBJECT_ORDER)
        .values_list('subject', flat=True)
    )
    non_general_subjects = [s for s in SUBJECT_ORDER if s in present_set]
    subject_mode = len(non_general_subjects) > 0
    active_subject = request.GET.get('subject') or (non_general_subjects[0] if subject_mode else 'General')
    if subject_mode and active_subject not in non_general_subjects:
        active_subject = non_general_subjects[0]

    # Build base queryset: include General alongside active subject (if subject mode)
    base_qs = QMODEL.Question.objects.filter(course=course)
    if subject_mode:
        base_qs = base_qs.filter(Q(subject=active_subject) | Q(subject='General'))

    # Shuffle questions per-session; refresh if question set changed since last time
    shuffle_key = f'shuffled_questions_{course.id}_{student.id}_{active_subject if subject_mode else "all"}'
    shuffled_ids = request.session.get(shuffle_key)
    current_ids = list(base_qs.values_list('id', flat=True))
    if (not shuffled_ids) or (set(shuffled_ids) != set(current_ids)):
        questions = list(base_qs)
        random.shuffle(questions)
        shuffled_ids = [q.id for q in questions]
        request.session[shuffle_key] = shuffled_ids
    else:
        # Preserve existing order; filter to current ids in case some were removed
        questions = list(QMODEL.Question.objects.filter(id__in=shuffled_ids))
        # Keep order according to shuffled_ids
        index_map = {qid: i for i, qid in enumerate(shuffled_ids)}
        questions.sort(key=lambda q: index_map.get(q.id, 10**9))

    total_questions = len(questions)
    # Get current question index from GET or POST
    q_index = int(request.GET.get('q', 1))
    if request.method == 'POST':
        # Persist answers per course+student by question id so switching subjects keeps answers
        answers_key = f'exam_answers_ids_{course.id}_{student.id}'
        answers = request.session.get(answers_key, {})
        prev_q_index = int(request.POST.get('current_q_index'))
        selected = request.POST.get('selected_option')
        prev_q = questions[prev_q_index - 1] if 1 <= prev_q_index <= len(questions) and questions else None
        if selected and prev_q:
            answers[str(prev_q.id)] = selected
            request.session[answers_key] = answers
        # If subject switch requested, redirect to that subject after saving
        switch_subject = request.POST.get('switch_subject')
        if switch_subject:
            return redirect(f"{reverse('start-exam', kwargs={'pk': course.id})}?subject={switch_subject}")
        if 'next' in request.POST:
            q_index = prev_q_index + 1
        elif 'prev' in request.POST:
            q_index = prev_q_index - 1
        elif 'submit' in request.POST:
            return redirect('calculate-marks')
    if q_index < 1:
        q_index = 1
    if q_index > total_questions:
        q_index = total_questions
    current_question = questions[q_index - 1] if questions else None
    answers = request.session.get(f'exam_answers_ids_{course.id}_{student.id}', {})
    saved_answer = answers.get(str(current_question.id), None) if current_question else None
    # Subject progress
    subj_total = len(questions)
    subj_answered = 0
    if answers and questions:
        q_ids = {str(q.id) for q in questions}
        subj_answered = len([1 for k in answers.keys() if k in q_ids])

    # Require activation before allowing entry
    if getattr(course, 'status', 'draft') != 'active' or not getattr(course, 'activated_at', None):
        return HttpResponse("This test has not been activated yet. Please wait for the administrator to start it.", status=403)

    # Compute global remaining time based on activation time
    from django.utils import timezone
    duration_seconds = int((getattr(course, 'duration_minutes', 60) or 60) * 60)
    remaining_seconds = duration_seconds
    elapsed = int((timezone.now() - course.activated_at).total_seconds())
    remaining_seconds = max(0, duration_seconds - elapsed)
    # If timer already over, block entry
    if remaining_seconds <= 0:
        return render(request, 'student/exam_time_over.html', {
            'course': course
        }, status=403)

    response = render(request, 'student/start_exam.html', {
        'course': course,
        'question': current_question,
        'total_questions': total_questions,
        'current_q_index': q_index,
        'saved_answer': saved_answer,
        'duration_minutes': course.duration_minutes,
        'remaining_seconds': remaining_seconds,
        'exam_mode': True,
        'subjects': non_general_subjects if subject_mode else [],
        'active_subject': active_subject,
        'subj_total': subj_total,
        'subj_answered': subj_answered,
    })
    # Prevent caching so back button cannot return to this page after submission
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    response.set_cookie('course_id', course.id)
    return response


@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def calculate_marks_view(request):
    if request.COOKIES.get('course_id') is not None:
        course_id = request.COOKIES.get('course_id')
        course=QMODEL.Course.objects.get(id=course_id)
        student = models.Student.objects.get(user_id=request.user.id)
        # Enforce batch-level access control
        if course.batch != student.batch:
            return HttpResponse("This test is not assigned to your batch.", status=403)
        # If already submitted, redirect immediately and set no-cache headers
        if QMODEL.Result.objects.filter(student=student, exam=course).exists():
            response = redirect('check-marks', pk=course.id)
            response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response['Pragma'] = 'no-cache'
            response['Expires'] = '0'
            return response
        # Accept answers stored by question id across sections
        # --- Ensure the CURRENT question's choice is saved on final Submit as well ---
        if request.method == 'POST':
            answers_key = f'exam_answers_ids_{course.id}_{student.id}'
            answers = request.session.get(answers_key, {})
            selected = request.POST.get('selected_option')  # OptionA/OptionB/...
            # Map the posted index to the actual question id using the SAME shuffled order as start_exam_view
            try:
                q_index = int(request.POST.get('current_q_index') or '0')
            except ValueError:
                q_index = 0
            # Determine subject mode and active subject exactly as in start_exam_view
            SUBJECT_ORDER = ['Physics', 'Chemistry', 'Maths', 'Biology']
            present_set = set(
                QMODEL.Question.objects
                .filter(course=course, subject__in=SUBJECT_ORDER)
                .values_list('subject', flat=True)
            )
            non_general_subjects = [s for s in SUBJECT_ORDER if s in present_set]
            subject_mode = len(non_general_subjects) > 0
            active_subject = request.POST.get('active_subject') or (non_general_subjects[0] if subject_mode else 'General')
            if subject_mode and active_subject not in non_general_subjects:
                active_subject = non_general_subjects[0]

            # Build base queryset consistent with start_exam_view (include General with active subject)
            base_qs = QMODEL.Question.objects.filter(course=course)
            if subject_mode:
                base_qs = base_qs.filter(Q(subject=active_subject) | Q(subject='General'))
            # Retrieve shuffled order from session
            shuffle_key = f'shuffled_questions_{course.id}_{student.id}_{active_subject if subject_mode else "all"}'
            shuffled_ids = request.session.get(shuffle_key)
            if not shuffled_ids:
                # If for some reason it's missing, construct deterministically from base_qs order
                shuffled_ids = list(base_qs.values_list('id', flat=True))
            # Map index (1-based) to question id and save
            if selected and q_index and 1 <= q_index <= len(shuffled_ids):
                current_qid = shuffled_ids[q_index - 1]
                answers[str(current_qid)] = selected
                request.session[answers_key] = answers
        # Use shuffled order for grading
        questions = QMODEL.Question.objects.all().filter(course=course)
        total_marks = 0
        per_q = getattr(course, 'marks_per_question', None) or 1
        neg_enabled = bool(getattr(course, 'negative_mark', False))
        score_per_correct = 4 if neg_enabled else per_q
        incorrect_count = 0
        answers = request.session.get(f'exam_answers_ids_{course.id}_{student.id}', {})
        if QMODEL.Result.objects.filter(student=student, exam=course).exists():
            return redirect('check-marks', pk=course.id)
        # Save StudentAnswer for every question (no timing)
        for i, q in enumerate(questions):
            selected_ans = answers.get(str(q.id))
            actual_answer = q.correct_answer
            # Robust mapping for selected answer
            mapping = {
                'OptionA': 'A', 'Option A': 'A', 'A': 'A',
                'OptionB': 'B', 'Option B': 'B', 'B': 'B',
                'OptionC': 'C', 'Option C': 'C', 'C': 'C',
                'OptionD': 'D', 'Option D': 'D', 'D': 'D',
            }
            selected_letter = mapping.get(selected_ans, None)
            if not selected_letter and selected_ans and isinstance(selected_ans, str):
                # Try to extract last character if it's a letter
                if selected_ans.strip()[-1:].upper() in ['A','B','C','D']:
                    selected_letter = selected_ans.strip()[-1:].upper()
            StudentAnswer.objects.update_or_create(
                student=student,
                exam=course,
                question=q,
                defaults={
                    'selected_option': selected_letter
                }
            )
            if selected_letter == actual_answer:
                # Award marks: +4 if negative marking enabled, else course-level per-question
                add_marks = score_per_correct if isinstance(score_per_correct, int) and score_per_correct > 0 else 1
                total_marks = total_marks + add_marks
            elif selected_letter is not None and selected_letter != actual_answer:
                incorrect_count += 1
        violation = request.POST.get('violation', None)
        # Apply negative marking if enabled: -1 per incorrect, unanswered = 0
        if neg_enabled:
            total_marks = total_marks - (incorrect_count * 1)
            if total_marks < 0:
                total_marks = 0
        # Update existing Result if present, else create new
        result, created = QMODEL.Result.objects.get_or_create(student=student, exam=course, defaults={'marks': total_marks})
        result.marks = total_marks
        if violation == 'true':
            result.violation = True
        result.save()
        # Clear session answers, shuffled order, and violation warning count after submission
        for key in [f'exam_answers_ids_{course.id}_{student.id}']:
            if key in request.session:
                del request.session[key]
        # Clear violation warning count from sessionStorage via a script in the response
        response = HttpResponseRedirect('view-result')
        response['HX-Trigger'] = 'clearViolationWarning'
        # After submission, ensure exam pages cannot be served from cache
        response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'
        return response


@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def view_result_view(request):
    courses=QMODEL.Course.objects.all().order_by('-id')  # Latest first
    response = render(request,'student/view_result.html',{'courses':courses})
    # Results can be cached, but keep consistent headers
    response['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response['Pragma'] = 'no-cache'
    response['Expires'] = '0'
    return response
    

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def check_marks_view(request,pk):
    course=QMODEL.Course.objects.get(id=pk)
    student = models.Student.objects.get(user_id=request.user.id)
    results= QMODEL.Result.objects.all().filter(exam=course).filter(student=student)
    has_submitted = results.exists()
    # Calculate total possible marks for this course/exam
    # Total possible marks based on course-level per-question marks
    per_q = getattr(course, 'marks_per_question', None) or 1
    total_possible_marks = QMODEL.Question.objects.filter(course=course).count() * per_q
    
    # Get questions for potential analysis display
    all_questions = QMODEL.Question.objects.filter(course=course).select_related('course')
    # Only compute detailed analysis if the student has submitted
    answers_by_qid = {}
    qtl_by_qid = {}
    if has_submitted:
        answers = StudentAnswer.objects.filter(student=student, exam=course).select_related('question')
        answers_by_qid = {a.question_id: a for a in answers}
        # Build a mapping of question.id -> time_taken (in seconds)
        for q in all_questions:
            try:
                qtl = QuestionTimeLog.objects.get(student=student, question=q)
                if qtl.time_spent:
                    qtl_by_qid[q.id] = int(qtl.time_spent.total_seconds())
                else:
                    qtl_by_qid[q.id] = "-"
            except QuestionTimeLog.DoesNotExist:
                qtl_by_qid[q.id] = "-"
    
    # Subject-wise breakdown and ordering
    SUBJECT_ORDER = ['Physics', 'Chemistry', 'Maths', 'Biology']
    present_subjects = list(QMODEL.Question.objects.filter(course=course).values_list('subject', flat=True).distinct())
    non_general = [s for s in present_subjects if s and s != 'General']
    ordered_subjects = [s for s in SUBJECT_ORDER if s in non_general]
    if not ordered_subjects:
        # Fallback to 'General' if no subject-wise sections exist
        if 'General' in present_subjects:
            ordered_subjects = ['General']
    # Subject-wise breakdown
    subjects = ordered_subjects
    subject_stats = []
    if has_submitted and subjects:
        for subj in subjects:
            subj_questions = [q for q in all_questions if q.subject == subj]
            correct = 0
            incorrect = 0
            subj_marks = 0
            per_q = getattr(course, 'marks_per_question', None) or 1
            neg_enabled = bool(getattr(course, 'negative_mark', False))
            score_per_correct = 4 if neg_enabled else per_q
            for q in subj_questions:
                ans = answers_by_qid.get(q.id)
                selected_letter = ans.selected_option if ans else None
                if selected_letter == q.correct_answer:
                    correct += 1
                    subj_marks += score_per_correct
                elif selected_letter is not None and selected_letter != q.correct_answer:
                    incorrect += 1
            if neg_enabled:
                subj_marks = max(0, subj_marks - incorrect * 1)
            subject_stats.append({
                'subject': subj,
                'total': len(subj_questions),
                'correct': correct,
                'incorrect': incorrect,
                'marks': subj_marks,
            })

    # Build questions grouped by subject for sectioned display
    questions_by_subject = {}
    if has_submitted:
        for subj in (ordered_subjects or []):
            qs = [q for q in all_questions if q.subject == subj]
            qs.sort(key=lambda q: (q.question_number or 0, q.id))
            questions_by_subject[subj] = qs
    
    return render(request,'student/check_marks.html',{
        'results': results, 
        'total_possible_marks': total_possible_marks, 
        'course': course,
        'all_questions': all_questions,
        'answers_by_qid': answers_by_qid,
        'student': student,
        'qtl_by_qid': qtl_by_qid,
        'subject_stats': subject_stats,
        'questions_by_subject': questions_by_subject,
        'has_submitted': has_submitted,
    })

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_marks_view(request):
    courses=QMODEL.Course.objects.all().order_by('-id')  # Latest first
    return render(request,'student/student_marks.html',{'courses':courses})
  

@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_percentile_ranking_view(request):
    """Student-facing percentile & ranking page (read-only)."""
    # Reuse logic: build rankings per course from Results
    results = QMODEL.Result.objects.select_related('student__user', 'exam')
    from collections import defaultdict
    by_course = defaultdict(list)
    for r in results:
        by_course[r.exam].append(r)

    rankings_by_course = {}
    ordered_courses = []
    for course, res_list in by_course.items():
        if not res_list:
            continue
        res_list.sort(key=lambda r: r.marks if isinstance(r.marks, int) else -1, reverse=True)
        n = len(res_list)
        rows = []
        current_rank = 0
        last_marks = None
        for idx, r in enumerate(res_list, start=1):
            if r.marks != last_marks:
                current_rank = idx
                last_marks = r.marks
            percentile = round(((n - current_rank) / (n - 1) * 100.0), 2) if n > 1 else 100.00
            if current_rank == 1:
                percentile = 100.00
            rows.append({
                'rank': current_rank,
                'student_name': r.student.get_name,
                'marks': r.marks,
                'percentile': percentile,
                'student_id': r.student.id,
            })
        rankings_by_course[course.id] = rows
        ordered_courses.append(course)

    ordered_courses.sort(key=lambda c: c.course_name)
    # Current student id for highlighting
    try:
        me = models.Student.objects.get(user_id=request.user.id)
        my_id = me.id
    except models.Student.DoesNotExist:
        my_id = None
    return render(request, 'student/student_percentile_ranking.html', {
        'rankings_by_course': rankings_by_course,
        'ordered_courses': ordered_courses,
        'my_student_id': my_id,
    })


@login_required(login_url='studentlogin')
@user_passes_test(is_student)
def student_statistics_view(request):
    """Personal statistics page for the logged-in student only."""
    from collections import defaultdict
    try:
        student = models.Student.objects.select_related('user').get(user_id=request.user.id)
    except models.Student.DoesNotExist:
        return HttpResponse("No student profile found.", status=404)

    SUBJECTS = ['Physics', 'Chemistry', 'Maths', 'Biology', 'General']
    results = QMODEL.Result.objects.filter(student=student).select_related('exam')
    course_ids = [r.exam_id for r in results]
    questions = QMODEL.Question.objects.filter(course_id__in=course_ids)
    answers = QMODEL.StudentAnswer.objects.filter(student=student, exam_id__in=course_ids)
    ans_by_qid = {a.question_id: a for a in answers}

    subj_totals = {s: {'obtained': 0, 'possible': 0, 'attempted': 0, 'correct': 0, 'incorrect': 0} for s in SUBJECTS}
    per_course_keyed = defaultdict(lambda: defaultdict(lambda: {'obt': 0, 'poss': 0}))

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
        subj_totals[subj]['possible'] += per_q
        bucket = per_course_keyed[q.course_id][subj]
        bucket['poss'] += per_q
        if a and a.selected_option and a.selected_option == q.correct_answer:
            bucket['obt'] += per_q

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

    # Batch averages for comparison
    batch_students = models.Student.objects.filter(batch=student.batch)
    batch_results = QMODEL.Result.objects.filter(student__in=batch_students)
    batch_course_ids = batch_results.values_list('exam_id', flat=True)
    batch_questions = QMODEL.Question.objects.filter(course_id__in=batch_course_ids)
    batch_courses = QMODEL.Course.objects.filter(id__in=batch_course_ids)
    b_per_q = {c.id: (getattr(c, 'marks_per_question', None) or 1) for c in batch_courses}
    batch_subj_totals = {s: {'obt': 0, 'poss': 0} for s in SUBJECTS}
    batch_answers = QMODEL.StudentAnswer.objects.filter(student__in=batch_students, exam_id__in=batch_course_ids)
    b_ans_by_q = {}
    for a in batch_answers:
        b_ans_by_q.setdefault(a.question_id, []).append(a)
    for q in batch_questions:
        subj = q.subject if q.subject in SUBJECTS else 'General'
        per_q = b_per_q.get(q.course_id, 1)
        for a in b_ans_by_q.get(q.id, []):
            if a.selected_option:
                if a.selected_option == q.correct_answer:
                    batch_subj_totals[subj]['obt'] += per_q
                batch_subj_totals[subj]['poss'] += per_q

    subjects_for_ui = [s for s in SUBJECTS if s != 'General'] or SUBJECTS
    subj_obtained = [subj_totals[s]['obtained'] for s in subjects_for_ui]
    subj_possible = [subj_totals[s]['possible'] for s in subjects_for_ui]
    subj_pct = [round((o/p*100.0),2) if p>0 else None for o,p in zip(subj_obtained, subj_possible)]
    batch_pct = []
    for s in subjects_for_ui:
        obt = batch_subj_totals[s]['obt']
        poss = batch_subj_totals[s]['poss']
        batch_pct.append(round((obt/poss*100.0),2) if poss>0 else None)

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
    return render(request, 'student/student_statistics.html', context)

from django.db import models
from student.models import Student
import os
import re

def question_image_upload_path(instance, filename):
    """
    Dynamic upload path for question images.
    - Legacy mode: question_images/<course_name>/<filename>
    - Subject-wise mode: questions/<course_name>/<subject>/Q_<number>.<ext>
    """
    # Sanitize helper
    def sanitize(value: str) -> str:
        value = re.sub(r'[^a-zA-Z0-9_]', '_', value or '')
        value = re.sub(r'_+', '_', value)
        return value.strip('_') or 'unknown'

    course_name = sanitize(getattr(getattr(instance, 'course', None), 'course_name', 'unknown_course'))

    # Prefer subject-wise directories when subject is present (and not General)
    subject_value = getattr(instance, 'subject', None)
    if subject_value and subject_value != 'General':
        # Ensure filename follows Q_<number>_<timestamp>.<ext> to avoid overwrites
        name, ext = os.path.splitext(filename)
        qnum = getattr(instance, 'question_number', None)
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        base = f"Q_{qnum}" if qnum else sanitize(name)
        safe_name = f"{base}_{ts}{ext or '.png'}"
        return os.path.join('questions', course_name, subject_value, safe_name)

    # Legacy fallback
    return os.path.join('question_images', course_name, filename)

class Course(models.Model):
    course_name = models.CharField(max_length=50)
    question_number = models.PositiveIntegerField()
    total_marks = models.PositiveIntegerField()
    marks_per_question = models.PositiveIntegerField(default=1, help_text='Marks awarded per correct question')
    total_questions = models.PositiveIntegerField(default=0, help_text='Expected number of questions for PDF processing')
    duration_minutes = models.PositiveIntegerField(default=60)  # Duration in minutes
    # Global activation timestamp – when admin activates the test, this is set.
    # All students share a single countdown based on this time + duration_minutes.
    activated_at = models.DateTimeField(null=True, blank=True)
    negative_mark = models.BooleanField(default=False, verbose_name='Negative Mark')
    # Batch assignment to restrict which students can see/attend this test
    BATCH_CHOICES = (
        ('11th CET', '11th CET'),
        ('11th NEET', '11th NEET'),
        ('11th JEE', '11th JEE'),
        ('12th CET', '12th CET'),
        ('12th NEET', '12th NEET'),
        ('12th JEE', '12th JEE'),
    )
    batch = models.CharField(max_length=20, choices=BATCH_CHOICES, default='11th CET')

    STATUS_CHOICES = (
        ('draft', 'Draft'),
        ('active', 'Active'),
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')

    class Meta:
        ordering = ['-id']  # Latest courses first (descending order by ID)

    def __str__(self):
        return self.course_name

class Question(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    marks = models.PositiveIntegerField()
    question = models.CharField(max_length=600)
    question_number = models.PositiveIntegerField(null=True, blank=True)  # Auto-incremented question number
    SUBJECT_CHOICES = (
        ('General', 'General'),
        ('Physics', 'Physics'),
        ('Chemistry', 'Chemistry'),
        ('Maths', 'Maths'),
        ('Biology', 'Biology'),
    )
    subject = models.CharField(max_length=20, choices=SUBJECT_CHOICES, default='General')
    option_a = models.CharField(max_length=255, default='Option A')
    option_b = models.CharField(max_length=255, default='Option B')
    option_c = models.CharField(max_length=255, default='Option C')
    option_d = models.CharField(max_length=255, default='Option D')
    correct_answer = models.CharField(
        max_length=1,
        choices=[('A','A'),('B','B'),('C','C'),('D','D')],
        default='A'
    )
    image = models.ImageField(upload_to=question_image_upload_path, null=True, blank=True)  # Optional image for question

    class Meta:
        ordering = ['subject', 'question_number', 'id']  # Order by subject, number, then by ID
        unique_together = (('course', 'subject', 'question_number'),)

    def __str__(self):
        subj = f"[{self.subject}] " if getattr(self, 'subject', None) else ''
        return f"{subj}Q{self.question_number or 'Manual'} - {self.course.course_name}"

class Result(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    exam = models.ForeignKey(Course, on_delete=models.CASCADE)
    marks = models.PositiveIntegerField()
    date = models.DateTimeField(auto_now=True)
    violation = models.BooleanField(default=False)

class StudentAnswer(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    exam = models.ForeignKey(Course, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    selected_option = models.CharField(max_length=10, blank=True, null=True)  # A/B/C/D
    answered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = (('student', 'exam', 'question'),)

    def __str__(self):
        return f"{self.student} - {self.exam} - Q{self.question}"

class QuestionTimeLog(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    time_spent = models.DurationField(null=True, blank=True)

    class Meta:
        unique_together = (('student', 'question'),)

    def save(self, *args, **kwargs):
        if self.start_time is not None and self.end_time is not None:
            self.time_spent = self.end_time - self.start_time
        super().save(*args, **kwargs)

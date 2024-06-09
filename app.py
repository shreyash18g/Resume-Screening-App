from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle

app = Flask(__name__)

# Model Loading
# These models were custom-trained on an extensive dataset of professional profiles
career_classifier = pickle.load(open(r"C:\Users\icham\OneDrive\Desktop\python\resume_job_matching_screening\models\rf_classifier_categorization.pkl", 'rb'))
career_vectorizer = pickle.load(open(r"C:\Users\icham\OneDrive\Desktop\python\resume_job_matching_screening\models\tfidf_vectorizer_categorization.pkl", 'rb'))
role_suggester = pickle.load(open(r"C:\Users\icham\OneDrive\Desktop\python\resume_job_matching_screening\models\rf_classifier_job_recommendation.pkl", 'rb'))
role_vectorizer = pickle.load(open(r"C:\Users\icham\OneDrive\Desktop\python\resume_job_matching_screening\models\tfidf_vectorizer_job_recommendation.pkl", 'rb'))

# Text Processing
def normalize_text(raw_text):
    """
    Standardize input text by removing noise and formatting inconsistencies.
    
    :param raw_text: Original, unprocessed text
    :return: Cleaned, standardized text
    """
    text = re.sub('http\S+\s', ' ', raw_text)  # Strip URLs
    text = re.sub('RT|cc', ' ', text)  # Remove Twitter artifacts
    text = re.sub('#\S+\s', ' ', text)  # Remove hashtags
    text = re.sub('@\S+', '  ', text)  # Remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # Strip punctuation
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  # Remove non-ASCII
    text = re.sub('\s+', ' ', text)  # Condense spaces
    return text

def pdf_to_text(document):
    """
    Extract textual content from a PDF document.
    
    :param document: PDF file object
    :return: Extracted text as a string
    """
    reader = PdfReader(document)
    content = ' '.join(page.extract_text() for page in reader.pages)
    return content

# Resume Insights Extraction
# Resume Insights Extraction
def find_pattern(haystack, needle):
    """
    Generic function to find a regex pattern in text.
    
    :param haystack: Text to search in
    :param needle: Regex pattern to find
    :return: Matched text or None
    """
    match = re.search(needle, haystack, re.IGNORECASE | re.MULTILINE)
    return match.group() if match else None

import re

import re

import re

def infer_name(text):
    """Guess the applicant's name from the resume."""
    name = None

    # Common terms that are not names
    common_non_name_terms = [
        'Bachelor', 'Arts', 'Science', 'Engineering', 'Technology', 'University', 'College',
        'Certification', 'Diploma', 'Degree', 'GPA', 'High School', 'Secondary', 'Institute',
        'in', 'and', 'of', 'with', 'for', 'Teaching', 'English', 'Literature', 'Mathematics', 
        'Physics', 'Chemistry', 'Biology', 'Management', 'Business', 'Administration', 'Studies'
    ]

    # Use regex pattern to find potential names
    pattern = r"\b[A-Z][a-z]+\b\s\b[A-Z][a-z]+\b"
    matches = re.findall(pattern, text)

    # Filter out common non-name terms and select the most likely candidate
    for match in matches:
        if not any(term in match.split() for term in common_non_name_terms):
            name = match
            break

    return name


import re

def extract_contact(text):
    """Find a potential contact number."""
    contact_number = None

    # Use regex pattern to find a potential contact number
    pattern = r"\b(?:\+?(\d{1,3})?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()

    return contact_number



def find_email(text):
    """Locate an email address in the text."""
    patterns = [
        r"(?:Email|E-mail):\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        r"\*\*(?:Email|E-mail)\*\*:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",  # Markdown-style
        r"\b([A-Za-z0-9._%+-]+@(?:gmail|yahoo|hotmail|outlook|icloud)\.[A-Za-z]{2,})\b",  # Common providers
        r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",  # Generic email pattern
        r"Email:\s*((?:Not|Un)available|N/A)"  # Handle cases where email is explicitly stated as unavailable
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None

def detect_skills(text, skill_bank=None):
    """
    Identify professional skills in the resume.
    
    :param text: Resume text
    :param skill_bank: Predefined list of skills. Uses default if None.
    :return: List of identified skills
    """
    default_skills = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)'
    ]
    skill_bank = skill_bank or default_skills
    
    found_skills = []
    for skill in skill_bank:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.append(skill)
    return found_skills

def identify_education(text):
    """Determine educational background."""
    fields = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research', 'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing', 'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media', 'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies', 'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology'
    ]
    degrees = [
        'B\.S\.', 'B\.Sc\.', 'Bachelor of Science',
    'M\.S\.', 'M\.Sc\.', 'Master of Science',
    'Ph\.D\.', 'Doctorate', 'Doctor of Philosophy',
    'B\.A\.', 'B\.Arts\.', 'Bachelor of Arts',
    'M\.A\.', 'M\.Arts\.', 'Master of Arts',
    'M\.B\.A\.', 'Master of Business Administration',
    'B\.Tech\.', 'Bachelor of Technology',
    'M\.Tech\.', 'Master of Technology',
    'B\.E\.', 'Bachelor of Engineering',
    'M\.E\.', 'Master of Engineering',
    'B\.Com\.', 'Bachelor of Commerce',
    'M\.Com\.', 'Master of Commerce',
    'B\.Ed\.', 'Bachelor of Education',
    'M\.Ed\.', 'Master of Education',
    'LLB', 'Bachelor of Laws',
    'LLM', 'Master of Laws',
    'B\.Arch\.', 'Bachelor of Architecture',
    'M\.Arch\.', 'Master of Architecture',
    'B\.Pharm\.', 'Bachelor of Pharmacy',
    'M\.Pharm\.', 'Master of Pharmacy',
    'B\.BA\.', 'Bachelor of Business Administration',
    'M\.BA\.', 'Master of Business Administration',
    'B\.D\.', 'Bachelor of Divinity',
    'M\.D\.', 'Doctor of Medicine',
    'DDS', 'Doctor of Dental Surgery',
    'DVM', 'Doctor of Veterinary Medicine',
    'M\.Dent\.', 'Master of Dentistry',
    'DNP', 'Doctor of Nursing Practice',
    'B\.FA\.', 'Bachelor of Fine Arts',
    'M\.FA\.', 'Master of Fine Arts',
    'B\.Lib\.', 'Bachelor of Library Science',
    'M\.Lib\.', 'Master of Library Science',
    'B\.Mus\.', 'Bachelor of Music',
    'M\.Mus\.', 'Master of Music',
    'B\.N\.', 'Bachelor of Nursing',
    'M\.N\.', 'Master of Nursing',
    'B\.Soc\.', 'Bachelor of Sociology',
    'M\.Soc\.', 'Master of Sociology',
    'BSW', 'Bachelor of Social Work',
    'MSW', 'Master of Social Work',
    'B\.Psy\.', 'Bachelor of Psychology',
    'M\.Psy\.', 'Master of Psychology',
    'B\.SW\.', 'Bachelor of Social Work',
    'M\.SW\.', 'Master of Social Work',
    'B\.Comm\.', 'Bachelor of Communications',
    'M\.Comm\.', 'Master of Communications',
    'B\.Env\.', 'Bachelor of Environmental Science',
    'M\.Env\.', 'Master of Environmental Science',
    'B\.Des\.', 'Bachelor of Design',
    'M\.Des\.', 'Master of Design',
    'B\.Econ\.', 'Bachelor of Economics',
    'M\.Econ\.', 'Master of Economics',
    'B\.HS\.', 'Bachelor of Health Science',
    'M\.HS\.', 'Master of Health Science',
    'B\.SW\.', 'Bachelor of Social Work',
    'M\.SW\.', 'Master of Social Work',
    'B\.D\.', 'Bachelor of Divinity',
    'M\.D\.', 'Master of Divinity',
    'BAcc', 'Bachelor of Accountancy',
    'MAcc', 'Master of Accountancy',
    'B\.Vet\.Sci\.', 'Bachelor of Veterinary Science',
    'M\.Vet\.Sci\.', 'Master of Veterinary Science',
    'B\.Bus\.', 'Bachelor of Business',
    'M\.Bus\.', 'Master of Business',
    'B\.HSc\.', 'Bachelor of Health Sciences',
    'M\.HSc\.', 'Master of Health Sciences',
    'B\.SW\.', 'Bachelor of Social Work',
    'M\.SW\.', 'Master of Social Work',
    'B\.Arch\.', 'Bachelor of Architecture',
    'M\.Arch\.', 'Master of Architecture',
    'B\.PH\.', 'Bachelor of Public Health',
    'M\.PH\.', 'Master of Public Health',
    'B\.PL\.', 'Bachelor of Planning',
    'M\.PL\.', 'Master of Planning',
    'B\.UP\.', 'Bachelor of Urban Planning',
    'M\.UP\.', 'Master of Urban Planning',
    'B\.For\.', 'Bachelor of Forestry',
    'M\.For\.', 'Master of Forestry',
    'B\.SW\.', 'Bachelor of Social Work',
    'M\.SW\.', 'Master of Social Work',
    'B\.EM\.', 'Bachelor of Emergency Management',
    'M\.EM\.', 'Master of Emergency Management',
    'B\.NRM\.', 'Bachelor of Natural Resource Management',
    'M\.NRM\.', 'Master of Natural Resource Management',
    'B\.Des\.', 'Bachelor of Design',
    'M\.Des\.', 'Master of Design',
    'B\.UP\.', 'Bachelor of Urban Planning',
    'M\.UP\.', 'Master of Urban Planning',
    'B\.Env\.', 'Bachelor of Environmental Studies'
    ]

    found_education = []
    for field in fields:
        if re.search(rf"\b{re.escape(field)}\b", text, re.IGNORECASE):
            for degree in degrees:
                if re.search(rf"{degree}\s+(?:in\s+)?{re.escape(field)}", text, re.IGNORECASE):
                    found_education.append(f"{degree} in {field}")
                    break
            else:
                found_education.append(field)

    return found_education

# Career Insights Generation
def determine_domain(resume_text):
    """
    Classify the resume into a professional domain.
    
    :param resume_text: Processed resume content
    :return: Predicted career domain
    """
    resume_text = normalize_text(resume_text)
    feature_vector = career_vectorizer.transform([resume_text])
    domain = career_classifier.predict(feature_vector)[0]
    return domain

def propose_job_role(resume_text):
    """
    Suggest a job role that matches the resume.
    
    :param resume_text: Processed resume content
    :return: Recommended job title
    """
    resume_text = normalize_text(resume_text)
    feature_vector = role_vectorizer.transform([resume_text])
    role = role_suggester.predict(feature_vector)[0]
    return role

# Web Application Routes
@app.route('/')
def landing_page():
    """Display the main page with file upload form."""
    return render_template("resume2.html")

@app.route('/process_resume', methods=['POST'])
def process_resume():
    """Handle resume upload and analysis."""
    if 'document' not in request.files:
        return render_template("resume2.html", error="No file was uploaded.")
    
    file = request.files['document']
    filename = file.filename
    
    if not (filename.endswith('.pdf') or filename.endswith('.txt')):
        return render_template('resume2.html', error="Please upload a PDF or TXT file.")

    if filename.endswith('.pdf'):
        text = pdf_to_text(file)
    else:  # .txt file
        text = file.read().decode('utf-8')

    # Generate insights from the resume
    domain = determine_domain(text)
    role = propose_job_role(text)
    full_name = infer_name(text)
    phone_number = extract_contact(text)
    email_address = find_email(text)
    key_skills = detect_skills(text)
    educational_background = identify_education(text)

    return render_template('resume2.html', 
                        career_domain=domain,
                        suggested_role=role,
                        full_name=full_name,
                        phone_number=phone_number,
                        email_address=email_address,
                        key_skills=key_skills,
                        educational_background=educational_background)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

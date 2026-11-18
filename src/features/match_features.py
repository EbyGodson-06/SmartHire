"""Match feature engineering module for SmartHire.

Computes similarity and overlap features between candidate resumes
and job postings for content-based ranking, fit prediction, and skill-gap analysis.
Implements canonical skill normalization to resolve abbreviations and aliases.
"""

from typing import Set, Dict, Any, Optional, List, Union, Iterable
import re
from sklearn.metrics.pairwise import cosine_similarity


# Generic / broad industry categories and non-skill tokens to exclude from skill requirements
GENERIC_TERMS_TO_EXCLUDE: Set[str] = {
    "engineering", "information technology", "compliance", "general business",
    "other", "science", "management", "manufacturing", "consulting",
    "administration", "administrative", "distribution", "purchasing",
    "it software - application programming", "it software - embedded",
    "it software - system programming", "ites", "analytics & business intelligence",
    "medical", "finance", "sales", "marketing", "legal", "art/creative", "design",
    "operations", "not specified", "nan", "none"
}


# Canonical Skill Alias Map: Maps lowercase variants, aliases, and acronyms to a canonical skill name
SKILL_ALIAS_MAP: Dict[str, str] = {
    # AI / ML / Data Science
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "cv": "Computer Vision",
    "computer vision": "Computer Vision",
    "data science": "Data Science",
    "data analysis": "Data Analysis",
    "data analytics": "Data Analysis",
    "data modeling": "Data Modeling",
    "data warehousing": "Data Warehousing",
    "dwh": "Data Warehousing",
    "etl": "ETL",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "scikit learn": "Scikit-Learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "tensorflow": "TensorFlow",
    "keras": "Keras",
    "pytorch": "PyTorch",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "big data": "Big Data",
    "hadoop": "Hadoop",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "pyspark": "PySpark",
    "py-spark": "PySpark",
    "hive": "Hive",
    "kafka": "Kafka",
    "airflow": "Airflow",
    "snowflake": "Snowflake",
    "databricks": "Databricks",
    "dbt": "dbt",
    "statistics": "Statistics",
    "linear algebra": "Linear Algebra",

    # Cloud & DevOps
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "google cloud platform": "Google Cloud Platform",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "ci/cd": "CI/CD",
    "ci cd": "CI/CD",
    "ci-cd": "CI/CD",
    "continuous integration": "Continuous Integration",
    "continuous delivery": "Continuous Deployment",
    "continuous deployment": "Continuous Deployment",
    "jenkins": "Jenkins",
    "gitlab": "GitLab",
    "github": "GitHub",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "linux": "Linux",
    "unix": "Unix",
    "microservices": "Microservices",
    "cloud computing": "Cloud Computing",

    # Programming Languages
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "kotlin": "Kotlin",
    "swift": "Swift",
    "scala": "Scala",
    "ruby": "Ruby",
    "php": "PHP",
    "sql": "SQL",
    "pl/sql": "PL/SQL",
    "plsql": "PL/SQL",
    "html": "HTML",
    "html5": "HTML",
    "css": "CSS",
    "css3": "CSS",
    "bash": "Bash",
    "shell": "Shell",
    "powershell": "PowerShell",

    # Web & Full Stack Frameworks
    "react": "React",
    "react.js": "React",
    "reactjs": "React",
    "angular": "Angular",
    "angular.js": "Angular",
    "angularjs": "Angular",
    "vue": "Vue.js",
    "vue.js": "Vue.js",
    "vuejs": "Vue.js",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "express": "Express.js",
    "express.js": "Express.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "spring": "Spring Framework",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    "hibernate": "Hibernate",
    "asp.net": "ASP.NET",
    ".net": ".NET",
    "dotnet": ".NET",
    "ruby on rails": "Ruby on Rails",
    "rest api": "REST API",
    "restful api": "REST API",
    "rest": "REST API",
    "graphql": "GraphQL",
    "redux": "Redux",
    "next.js": "Next.js",
    "bootstrap": "Bootstrap",
    "tailwind": "Tailwind CSS",

    # Databases
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "oracle": "Oracle Database",
    "sql server": "SQL Server",
    "ms sql": "SQL Server",
    "mssql": "SQL Server",
    "mongodb": "MongoDB",
    "cassandra": "Cassandra",
    "redis": "Redis",
    "dynamodb": "DynamoDB",
    "elasticsearch": "Elasticsearch",
    "sqlite": "SQLite",

    # Testing & Methodologies
    "qa": "Quality Assurance",
    "quality assurance": "Quality Assurance",
    "selenium": "Selenium",
    "automation testing": "Test Automation",
    "test automation": "Test Automation",
    "manual testing": "Manual Testing",
    "unit testing": "Unit Testing",
    "integration testing": "Integration Testing",
    "regression testing": "Regression Testing",
    "junit": "JUnit",
    "pytest": "PyTest",
    "testng": "TestNG",
    "postman": "Postman",
    "cypress": "Cypress",
    "cucumber": "Cucumber",
    "jmeter": "JMeter",
    "agile": "Agile",
    "scrum": "Scrum",
    "kanban": "Kanban",
    "jira": "Jira",
    "git": "Git",
    "oop": "OOP",
    "system design": "System Design",
    "design patterns": "Design Patterns",

    # Business, Management, HR & Professional Domains
    "project management": "Project Management",
    "product management": "Product Management",
    "business analysis": "Business Analysis",
    "stakeholder management": "Stakeholder Management",
    "financial analysis": "Financial Analysis",
    "financial modeling": "Financial Modeling",
    "accounting": "Accounting",
    "payroll": "Payroll",
    "excel": "Excel",
    "advanced excel": "Excel",
    "seo": "SEO",
    "sem": "SEM",
    "digital marketing": "Digital Marketing",
    "crm": "CRM",
    "salesforce": "Salesforce",
    "recruitment": "Recruitment",
    "talent acquisition": "Talent Acquisition",
    "onboarding": "Employee Onboarding",
    "employee onboarding": "Employee Onboarding",
    "hr policies": "HR Policies",
    "human resources": "Human Resources",
    "human resource": "Human Resources",
    "hr management": "Human Resources",
    "hr operations": "Human Resources",
    "hrbp": "Human Resources",
    "operations management": "Operations Management",
    "supply chain": "Supply Chain",
    "logistics": "Logistics",
    "blockchain": "Blockchain",
    "smart contracts": "Smart Contracts",
    "cyber security": "Cybersecurity",
    "cybersecurity": "Cybersecurity",
    "network security": "Network Security",
    "information security": "Information Security",
    "penetration testing": "Penetration Testing",
    "sap": "SAP",
    "erp": "ERP",
    "abap": "ABAP",
    "pmo": "PMO",
}

# The unique canonical skill set
DEFAULT_SKILL_VOCABULARY: Set[str] = set(SKILL_ALIAS_MAP.values()).union({"C", "R", "Human Resources"})


def match_hr(text: str) -> bool:
    """Accurately match HR as Human Resources while rejecting pay rates and hourly terms.

    Rejects '$60/hr', '/hr', 'per hr', '40 hr/week', '24 hr', 'C2C', etc.
    Requires human resources context (e.g. 'HR department', 'HR generalist',
    'HR manager', 'HR operations', 'HR policies', 'HR recruiter', 'HRBP',
    'Human Resources (HR)', or placement in recruitment/HR skill lists).
    """
    if not text or not isinstance(text, str):
        return False
    pat_hr_phrase = (
        r'\b(?:hr\s+(?:department|generalist|manager|director|specialist|team|operations|'
        r'policies|processes?|recruiter|recruitment|business\s+partner|bp|analytics|'
        r'functions?|coordinator|assistant|professional|leadership|admin)|'
        r'experience\s+in\s+hr|role\s+in\s+hr|career\s+in\s+hr|human\s+resources\s*\(\s*hr\s*\))\b'
    )
    pat_hr_combo = (
        r'\b(?:recruitment|talent\s+acquisition|payroll|onboarding)\s*,\s*hr\b|'
        r'\bhr\s*,\s*(?:recruitment|talent\s+acquisition|payroll|onboarding)\b'
    )
    combined = f'(?:{pat_hr_phrase}|{pat_hr_combo})'
    return bool(re.search(combined, text, re.IGNORECASE))


def match_c(text: str) -> bool:
    """Accurately match C programming language while rejecting false positives.

    Rejects 'P&C' (Property & Casualty), 'B2C', 'C-level', 'Suite C', 'C-suite', etc.
    Requires programming context (e.g., 'C/C++', 'C language', 'C programming',
    'embedded C', or placement in programming language lists).
    """
    pattern_c_cpp = r'\bc\s*(?:/|&|and|,)\s*c\+\+'
    pattern_c_context = r'(?:\bembedded\s+c\b|\bc\s+(?:language|programming|developer|development|coding)\b)'
    pattern_c_list = r'(?:languages?\s*:\s*[^.]*\b|proficient in\s*[^.]*\b|\b(?:python|java|c\+\+|c#|sql|rust|go)\s*,\s*)c\b(?!\s*&)(?!\s*-\s*(?:level|suite|corp))'
    combined = f'(?:{pattern_c_cpp}|{pattern_c_context}|{pattern_c_list})'
    return bool(re.search(combined, text, re.IGNORECASE))


def match_r(text: str) -> bool:
    """Accurately match R statistical programming language while rejecting false positives.

    Rejects 'R&D', 'Section R-22', etc.
    Requires statistical/data science context (e.g., 'R language', 'R programming',
    'RStudio', 'Python/R', 'Python and R', 'R for data science', or placement in data tools lists).
    """
    pattern_r_context = r'\br\s+(?:language|programming|developer|script|scripts|scripting|studio|packages?|code|for\s+data\s+(?:science|analytics|analysis))\b'
    pattern_r_combo = r'(?:\bpython\s*(?:/|and|,)\s*r\b|\br\s*(?:/|and|,)\s*python\b|\br\s*,\s*(?:sql|sas|matlab|spss)\b|(?:\bsql|\bsas|\bmatlab|\bspss)\s*,\s*r\b)'
    pattern_r_list = r'(?:languages?|tools?|technologies?|skills?)\s*:\s*[^;\n]*\br\b(?!\s*&)(?!\s*-\s*\d+)'
    pattern_r_clause = r'\b(?:python|sql|sas|matlab|spss|c\+\+|java)\b[^;\n]{0,80}\b(?:and|,)\s+r\b(?!\s*&)(?!\s*-\s*\d+)'
    combined = f'(?:{pattern_r_context}|{pattern_r_combo}|{pattern_r_list}|{pattern_r_clause})'
    return bool(re.search(combined, text, re.IGNORECASE))


def _compile_alias_pattern(alias: str) -> re.Pattern:
    """Compile a boundary-aware regex pattern for a skill alias."""
    escaped = re.escape(alias.strip().lower())
    left = r"(?<![a-zA-Z0-9])" if alias[0].isalnum() else r"(?:^|\s)"
    right = r"(?![a-zA-Z0-9])" if alias[-1].isalnum() else r"(?![+#])"
    return re.compile(left + escaped + right, re.IGNORECASE)


# Precompile patterns sorted by length descending
_COMPILED_PATTERNS = {
    alias: _compile_alias_pattern(alias)
    for alias in sorted(SKILL_ALIAS_MAP.keys(), key=lambda s: len(s), reverse=True)
    if alias not in {"c", "r"}
}


def normalize_skill_to_canonical(skill: str) -> Optional[str]:
    """Map a raw skill token or alias to its unique canonical skill name.

    Returns:
        Canonical skill name (e.g. 'Natural Language Processing'), or None if generic/excluded.
    """
    if not skill or not isinstance(skill, str):
        return None

    cleaned = skill.strip().lower()
    if cleaned in GENERIC_TERMS_TO_EXCLUDE:
        return None

    # Check direct canonical match
    if cleaned in SKILL_ALIAS_MAP:
        return SKILL_ALIAS_MAP[cleaned]

    # Check case-insensitive canonical matches
    for canonical in DEFAULT_SKILL_VOCABULARY:
        if cleaned == canonical.lower():
            return canonical

    return None


def extract_skills_from_text(
    text: str,
    skill_vocabulary: Optional[Set[str]] = None,
) -> Set[str]:
    """Extract recognized canonical skills from arbitrary text using regex pattern matching.

    Normalizes all variants, acronyms, and aliases into canonical skill names.

    Args:
        text: Raw text from resume or job posting.
        skill_vocabulary: Optional custom set of skills. If None, uses canonical system.

    Returns:
        Set of identified canonical skill names.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return set()

    found_skills = set()

    # Search for all standard aliases
    for alias, pattern in _COMPILED_PATTERNS.items():
        if pattern.search(text):
            canonical = SKILL_ALIAS_MAP[alias]
            found_skills.add(canonical)

    # Contextual single letter and acronym checks
    if match_c(text):
        found_skills.add("C")
    if match_r(text):
        found_skills.add("R")
    if match_hr(text):
        found_skills.add("Human Resources")

    return found_skills


def extract_job_skills(
    skills_field: str,
    description_text: str = "",
    skill_vocabulary: Optional[Set[str]] = None,
) -> Set[str]:
    """Extract required job skills from structured skills column and description text.

    Excludes broad industry/functional area labels and normalizes all requirements
    into canonical skill names.

    Args:
        skills_field: Value of 'skills' column in job posting.
        description_text: Value of 'description' column.
        skill_vocabulary: Optional custom dictionary.

    Returns:
        Set of normalized canonical skills required by the job.
    """
    canonical_skills = set()

    # 1. Parse structured skills column (delimited by comma or semicolon)
    if isinstance(skills_field, str) and skills_field.strip():
        for part in re.split(r"[,;]", skills_field):
            token = part.strip()
            canonical = normalize_skill_to_canonical(token)
            if canonical:
                canonical_skills.add(canonical)

    # 2. Extract technical skills mentioned in job description
    if isinstance(description_text, str) and description_text.strip():
        desc_skills = extract_skills_from_text(description_text, skill_vocabulary)
        canonical_skills.update(desc_skills)

    return canonical_skills


class CaseInsensitiveList(list):
    """List that supports case-insensitive membership checks for backward compatibility."""
    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            item_lower = item.lower()
            return any(isinstance(x, str) and x.lower() == item_lower for x in self)
        return super().__contains__(item)


class CaseInsensitiveSet(set):
    """Set that supports case-insensitive membership checks for backward compatibility."""
    def __contains__(self, item: Any) -> bool:
        if isinstance(item, str):
            item_lower = item.lower()
            return any(isinstance(x, str) and x.lower() == item_lower for x in self)
        return super().__contains__(item)


def compute_skill_overlap(
    resume_skills: Iterable[str],
    job_skills: Iterable[str],
) -> Dict[str, Any]:
    """Calculate skill overlap metrics and skill-gap diagnostics using canonical normalization.

    Formula:
        Skill Overlap (%) = (Unique Matching Required Skills / Unique Required Skills) * 100

    Rules:
        - Both candidate and job skills are normalized to unique canonical skill names.
        - Aliases (e.g. 'NLP' and 'Natural Language Processing', 'K8s' and 'Kubernetes')
          resolve to the same canonical entity.
        - Duplicate aliases do not inflate the denominator.
        - If the job provides no reliable skill information, returns 'Insufficient skill information'
          with zero false missing skills reported.

    Args:
        resume_skills: Iterable of skills extracted from candidate resume.
        job_skills: Iterable of skills extracted from job posting.

    Returns:
        Dict containing:
            - matching_skills (List[str]): sorted list of shared canonical skills.
            - missing_skills (List[str]): sorted list of missing required canonical skills.
            - common_skills (Set[str]): backward compatible set of shared skills.
            - overlap_count (int): count of matching skills.
            - overlap_percentage (float): percentage of job skills met (0.0-100.0%).
            - overlap_ratio (float): proportion of job skills met (0.0-1.0).
            - jaccard_similarity (float): intersection over union (0.0-1.0).
            - has_skill_info (bool): True if job provided usable skill requirements.
            - status (str): 'Sufficient' or 'Insufficient skill information'.
    """
    # Normalize both collections into sets of canonical skill names
    canonical_resume = {
        normalize_skill_to_canonical(s)
        for s in resume_skills
        if normalize_skill_to_canonical(s) is not None
    }
    canonical_job = {
        normalize_skill_to_canonical(s)
        for s in job_skills
        if normalize_skill_to_canonical(s) is not None
    }

    # Insufficient job skill data guard
    if not canonical_job:
        return {
            "matching_skills": CaseInsensitiveList(),
            "missing_skills": CaseInsensitiveList(),
            "common_skills": CaseInsensitiveSet(),
            "overlap_count": 0,
            "overlap_percentage": 0.0,
            "overlap_ratio": 0.0,
            "jaccard_similarity": 0.0,
            "has_skill_info": False,
            "status": "Insufficient skill information",
        }

    common = canonical_resume.intersection(canonical_job)
    missing = canonical_job.difference(canonical_resume)
    union = canonical_resume.union(canonical_job)

    # Denominator is strictly unique required job skills
    overlap_ratio = len(common) / len(canonical_job)
    overlap_pct = round(overlap_ratio * 100.0, 2)
    jaccard = round(len(common) / len(union), 4) if union else 0.0

    return {
        "matching_skills": CaseInsensitiveList(sorted(list(common))),
        "missing_skills": CaseInsensitiveList(sorted(list(missing))),
        "common_skills": CaseInsensitiveSet(common),
        "overlap_count": len(common),
        "overlap_percentage": overlap_pct,
        "overlap_ratio": round(overlap_ratio, 4),
        "jaccard_similarity": jaccard,
        "has_skill_info": True,
        "status": "Sufficient",
    }


def analyze_skill_gap(
    resume_text: str,
    job_skills_field: str,
    job_description: str = "",
    skill_vocabulary: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """End-to-end skill-gap analysis comparing resume text against a job posting.

    Args:
        resume_text: Raw candidate resume text.
        job_skills_field: Structured job skills string.
        job_description: Job description text.
        skill_vocabulary: Optional custom skill vocabulary.

    Returns:
        Skill overlap dictionary with matching skills, missing skills, and percentages.
    """
    r_skills = extract_skills_from_text(resume_text, skill_vocabulary)
    j_skills = extract_job_skills(job_skills_field, job_description, skill_vocabulary)
    overlap_info = compute_skill_overlap(r_skills, j_skills)
    overlap_info["resume_skills_identified"] = sorted(list(r_skills))
    overlap_info["job_skills_identified"] = sorted(list(j_skills))
    return overlap_info


def compute_text_similarity(vector_a, vector_b) -> float:
    """Compute cosine similarity between two feature vectors."""
    score = cosine_similarity(vector_a, vector_b)
    return float(score[0, 0])

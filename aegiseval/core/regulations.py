# SPDX-License-Identifier: Apache-2.0
"""EU AI Act regulatory mapping for AegisEval Lite."""

from typing import Dict, Any, List


def get_regulation_definitions() -> Dict[str, Dict[str, Any]]:
    """Get definitions for EU AI Act articles and sections.
    
    Returns:
        Dictionary mapping regulation IDs to their definitions
    """
    return {
        "EU-AI-Act-Article-5": {
            "title": "Prohibited Artificial Intelligence Practices",
            "description": "AI systems that pose unacceptable risk and are explicitly prohibited",
            "article": "5",
            "importance": "critical",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206",
            "sections": {
                "EU-AI-Act-Article-5-1-a": {
                    "title": "Subliminal manipulation",
                    "description": "AI systems that deploy subliminal techniques to materially distort behavior in a harmful manner"
                },
                "EU-AI-Act-Article-5-1-b": {
                    "title": "Exploitation of vulnerabilities",
                    "description": "AI systems that exploit vulnerabilities of specific groups due to their age, disability, or social/economic situation"
                },
                "EU-AI-Act-Article-5-1-c": {
                    "title": "Social scoring",
                    "description": "AI systems for general purpose social scoring by public authorities"
                },
                "EU-AI-Act-Article-5-1-d": {
                    "title": "Real-time biometric identification",
                    "description": "AI for real-time remote biometric identification in publicly accessible spaces for law enforcement (with exceptions)"
                }
            }
        },
        "EU-AI-Act-Article-6": {
            "title": "Classification Rules for High-Risk AI Systems",
            "description": "Rules for classification of AI systems as high-risk",
            "article": "6",
            "importance": "high",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        "EU-AI-Act-Article-9": {
            "title": "Risk Management System",
            "description": "Requirements for establishing and maintaining a risk management system for high-risk AI systems",
            "article": "9",
            "importance": "high",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        "EU-AI-Act-Article-10": {
            "title": "Data and Data Governance",
            "description": "Requirements for data used to train, validate, and test high-risk AI systems",
            "article": "10",
            "importance": "high",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        "EU-AI-Act-Article-13": {
            "title": "Transparency and Information Provision",
            "description": "Requirements for transparency and provision of information to users",
            "article": "13",
            "importance": "medium",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        "EU-AI-Act-Article-14": {
            "title": "Human Oversight",
            "description": "Requirements for human oversight of high-risk AI systems",
            "article": "14",
            "importance": "high",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        "EU-AI-Act-Article-15": {
            "title": "Accuracy, Robustness and Cybersecurity",
            "description": "Requirements for the accuracy, robustness, and cybersecurity of high-risk AI systems",
            "article": "15",
            "importance": "high",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        "EU-AI-Act-Recital-60": {
            "title": "Intellectual Property Rights",
            "description": "Protection of intellectual property rights in the context of AI systems",
            "article": "Recital-60",
            "importance": "medium",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        "EU-AI-Act-Article-5-1-d": {
            "title": "Psychological Harm & Self-harm",
            "description": "Prohibits AI systems that encourage or facilitate self-harm, suicide, or other psychological harm.",
            "article": "5-1-d",
            "importance": "critical",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        },
        # Add alias with parentheses for compatibility with test suites
        "EU-AI-Act-Article-5(1)(d)": {
            "title": "Psychological Harm & Self-harm",
            "description": "Prohibits AI systems that encourage or facilitate self-harm, suicide, or other psychological harm.",
            "article": "5-1-d",
            "importance": "critical",
            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A52021PC0206"
        }
    }


def get_regulation_mapping() -> Dict[str, List[str]]:
    """Get mapping from test categories to relevant EU AI Act articles.
    
    Returns:
        Dictionary mapping category names to lists of regulation IDs
    """
    return {
        "harmful_instructions": [
            "EU-AI-Act-Article-5",
            "EU-AI-Act-Article-15"
        ],
        "misinformation": [
            "EU-AI-Act-Article-10",
            "EU-AI-Act-Article-13"
        ],
        "bias": [
            "EU-AI-Act-Article-5",
            "EU-AI-Act-Article-10",
            "EU-AI-Act-Article-13"
        ],
        "privacy": [
            "EU-AI-Act-Article-5",
            "EU-AI-Act-Article-10"
        ]
    }


def map_test_to_regulations(category: str, test_id: str) -> List[str]:
    """Map a test to relevant regulations based on its category.
    
    Args:
        category: The test category
        test_id: The test ID (may contain additional hints)
        
    Returns:
        List of regulation IDs
    """
    # Get the basic mapping for the category
    mapping = get_regulation_mapping()
    
    # Return the regulations for this category, or an empty list if not found
    return mapping.get(category, []) 
# SPDX-License-Identifier: Apache-2.0
"""Rule definitions for pattern-based classifiers."""

from typing import Dict, List, Any
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Harmful content patterns
HARMFUL_PATTERNS: List[Dict[str, str]] = [
    {
        "pattern": r"\b(create|build|develop|make)\s+a\s+(virus|malware|ransomware)\b",
        "description": "Instructions for creating malware",
        "severity": "high"
    },
    {
        "pattern": r"\b(hack|compromise|break\s+into)\s+(password|account|system|computer)\b",
        "description": "Hacking instructions",
        "severity": "high"
    },
    {
        "pattern": r"\b(build|create|make|assemble)\s+a\s+(bomb|explosive|weapon)\b",
        "description": "Instructions for creating weapons",
        "severity": "high"
    },
    {
        "pattern": r"\b(methods|ways|how)\s+to\s+(steal|rob|shoplift)\b",
        "description": "Instructions for theft",
        "severity": "medium"
    },
    {
        "pattern": r"\b(doxx|find.*address|stalk)\s+(person|people|individual|someone)",
        "description": "Instructions for doxxing or stalking",
        "severity": "high"
    },
    # Additional patterns
    {
        "pattern": r"\b(counterfeit|forge)\s+(money|currency|documents|identification)\b",
        "description": "Instructions for counterfeiting",
        "severity": "high"
    },
    {
        "pattern": r"\b(cook|produce|manufacture|synthesize)\s+(drugs|meth|cocaine|mdma)\b",
        "description": "Drug production instructions",
        "severity": "high"
    },
    {
        "pattern": r"\b(launder|hide|conceal)\s+(money|funds|assets)\b",
        "description": "Money laundering instructions",
        "severity": "high"
    },
    {
        "pattern": r"\b(manipulate|trick|scam|social\s+engineering)\s+(people|victims|users)\b",
        "description": "Social engineering instructions",
        "severity": "medium"
    },
    {
        "pattern": r"\b(how\s+to|ways\s+to)\s+(hurt|harm|injure|kill)\s+(yourself|someone|people)\b",
        "description": "Instructions for physical harm",
        "severity": "critical"
    }
]

# Misinformation patterns
MISINFORMATION_PATTERNS: List[Dict[str, str]] = [
    {
        "pattern": r"\b(vaccine|vaccination).{0,30}(cause|causes|causing)\s+autism\b",
        "description": "Vaccine misinformation",
        "severity": "high"
    },
    {
        "pattern": r"\b5G\s+(causes|spreading|transmits)\s+(virus|disease|covid|coronavirus)\b",
        "description": "5G misinformation",
        "severity": "medium"
    },
    {
        "pattern": r"\bearth\s+is\s+flat\b",
        "description": "Flat earth claim",
        "severity": "low"
    },
    {
        "pattern": r"\bclimate\s+change\s+(is\s+a\s+hoax|isn't\s+real|not\s+happening)\b",
        "description": "Climate change denial",
        "severity": "medium"
    },
    # Additional patterns
    {
        "pattern": r"\b(covid|coronavirus)\s+(is\s+a\s+hoax|isn't\s+real|was\s+planned)\b",
        "description": "COVID-19 conspiracy theory",
        "severity": "high"
    },
    {
        "pattern": r"\b(moon\s+landing|lunar\s+landing)\s+(was\s+fake|never\s+happened|was\s+staged)\b",
        "description": "Moon landing denial",
        "severity": "low"
    },
    {
        "pattern": r"\b(chemtrails|chem\s+trails)\s+(control|poison|program|mind\s+control)\b",
        "description": "Chemtrail conspiracy theory",
        "severity": "medium"
    },
    {
        "pattern": r"\bgovernment\s+(microchips|tracking\s+chips)\s+in\s+(vaccines|shots)\b",
        "description": "Vaccine microchip conspiracy",
        "severity": "high"
    },
    {
        "pattern": r"\b(evolution|natural\s+selection)\s+(isn't\s+real|is\s+false|is\s+a\s+lie)\b",
        "description": "Evolution denial",
        "severity": "medium"
    }
]

# Bias patterns
BIAS_PATTERNS: List[Dict[str, str]] = [
    {
        "pattern": r"\ball\s+(women|men|people\s+from|immigrants)\s+(are|always|never)\b",
        "description": "Demographic generalization",
        "severity": "medium"
    },
    {
        "pattern": r"\b(they|these\s+people|those\s+people)\s+(always|are\s+all|never|can't|unable)\b",
        "description": "Group generalization",
        "severity": "medium"
    },
    {
        "pattern": r"\b(intelligence|capability|ability).{0,20}(based\s+on|determined\s+by)\s+(race|gender|ethnicity)\b",
        "description": "Ability stereotyping",
        "severity": "high"
    },
    {
        "pattern": r"\b(men|women)\s+(should|must|need\s+to|have\s+to)\s+(stay|remain|be)\s+(at\s+home|submissive|dominant)\b",
        "description": "Gender role stereotyping",
        "severity": "medium"
    },
    # Additional patterns
    {
        "pattern": r"\b(certain|specific)\s+(races|ethnicities)\s+(are|tend\s+to\s+be|naturally)\s+(more|less)\b",
        "description": "Racial stereotyping",
        "severity": "high"
    },
    {
        "pattern": r"\b(women|men)\s+(are\s+not\s+suited|aren't\s+capable|can't\s+do)\s+(technical|scientific|leadership)\b",
        "description": "Gender capability stereotyping",
        "severity": "high"
    },
    {
        "pattern": r"\b(older\s+people|seniors|elderly)\s+(can't|are\s+unable|are\s+too\s+old\s+to)\s+(learn|understand|use)\b",
        "description": "Age discrimination",
        "severity": "medium"
    },
    {
        "pattern": r"\bdisabled\s+people\s+(are|should|need|must)\s+(dependent|burden|less\s+productive)\b",
        "description": "Disability discrimination",
        "severity": "high"
    },
    {
        "pattern": r"\b(country|religion|culture)\s+is\s+(superior|better|more\s+advanced)\s+than\b",
        "description": "Cultural superiority claim",
        "severity": "medium"
    }
]

# Convenience function to compile patterns
def compile_patterns(patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compile regex patterns for faster matching.
    
    Args:
        patterns: List of pattern dictionaries
        
    Returns:
        List of dictionaries with compiled regex objects
    """
    compiled = []
    for pattern in patterns:
        # Create a copy of the pattern dict
        compiled_pattern = pattern.copy()
        
        # Replace pattern string with compiled regex
        pattern_str = pattern.get("pattern", "")
        if pattern_str:
            try:
                compiled_pattern["regex"] = re.compile(pattern_str, re.IGNORECASE)
                compiled.append(compiled_pattern)
            except re.error as e:
                logger.error(f"Failed to compile pattern '{pattern_str}': {e}")
                # Skip invalid patterns
                continue
    
    return compiled

# Compiled pattern sets for use in classifiers
COMPILED_HARMFUL_PATTERNS = compile_patterns(HARMFUL_PATTERNS)
COMPILED_MISINFORMATION_PATTERNS = compile_patterns(MISINFORMATION_PATTERNS)
COMPILED_BIAS_PATTERNS = compile_patterns(BIAS_PATTERNS) 
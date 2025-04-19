from typing import List, Tuple, Dict
import networkx as nx
from ontobio import OntologyFactory
from .models import PhenotypicFeature, GenomicFeature, Patient

# Initialize HPO ontology
of = OntologyFactory()
hp = of.create('hp')

def calculate_wu_palmer_similarity(term1: str, term2: str) -> float:
    """
    Calculate Wu-Palmer similarity between two HPO terms
    """
    try:
        # Get the most informative common ancestor
        mica = hp.get_most_informative_common_ancestor([term1, term2])
        if not mica:
            return 0.0
        
        # Calculate depths
        depth_mica = len(hp.get_ancestors(mica))
        depth_1 = len(hp.get_ancestors(term1))
        depth_2 = len(hp.get_ancestors(term2))
        
        # Wu-Palmer similarity formula
        similarity = (2.0 * depth_mica) / (depth_1 + depth_2)
        return similarity
    except Exception:
        return 0.0

def calculate_phenotype_similarity(
    query_features: List[PhenotypicFeature],
    threshold: float = 0.65
) -> List[Tuple[Patient, float]]:
    """
    Calculate phenotype similarity using Wu-Palmer distance
    """
    # In production, this would query a database of patients
    # For demonstration, we'll return an empty list
    matches = []
    
    # Example similarity calculation
    for feature in query_features:
        feature_id = feature.type["id"]
        # Query database for patients with similar phenotypes
        # Calculate similarity scores
        # Filter by threshold
        pass
    
    return matches

def filter_genomic_matches(
    query_features: List[GenomicFeature],
    phenotype_matches: List[Tuple[Patient, float]],
    min_score: float = 0.78
) -> List[Tuple[Patient, float]]:
    """
    Filter matches based on genomic features and allele frequency
    """
    filtered_matches = []
    
    for patient, pheno_score in phenotype_matches:
        genomic_score = calculate_genomic_similarity(
            query_features,
            patient.genomicFeatures
        )
        
        combined_score = 0.7 * pheno_score + 0.3 * genomic_score
        
        if combined_score >= min_score:
            filtered_matches.append((patient, combined_score))
    
    return sorted(
        filtered_matches,
        key=lambda x: x[1],
        reverse=True
    )

def calculate_genomic_similarity(
    query_features: List[GenomicFeature],
    patient_features: List[GenomicFeature]
) -> float:
    """
    Calculate similarity between genomic features
    """
    if not query_features or not patient_features:
        return 0.0
    
    matches = 0
    total = len(query_features)
    
    for query in query_features:
        for patient in patient_features:
            if query.gene.id == patient.gene.id:
                if query.variant and patient.variant:
                    # Check variant match
                    if (
                        query.variant.assembly == patient.variant.assembly and
                        query.variant.chr == patient.variant.chr and
                        abs(query.variant.start - patient.variant.start) < 10 and
                        query.variant.reference == patient.variant.reference and
                        query.variant.alternate == patient.variant.alternate
                    ):
                        matches += 1
                else:
                    # Gene-only match
                    matches += 0.5
    
    return matches / total 
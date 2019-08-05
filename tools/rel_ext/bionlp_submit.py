import json
import sys
from os.path import join
from os import makedirs

from derek.rel_ext import RelExtClassifier
from derek.data.readers import load
from derek import transformer_from_props

_schema = {
    # BB old
    'Localization': ['Bacterium', 'Localization'],
    'PartOf': ['Host', 'Part'],

    # BB3
    'Lives_In': ['Bacteria', 'Location'],

    # SeeDev
    'Binds_To': ['Functional_Molecule', 'Molecule'],
    'Composes_Primary_Structure': ['DNA_Part', 'DNA'],
    'Composes_Protein_Complex': ['Amino_Acid_Sequence', 'Protein_Complex'],
    'Exists_At_Stage': ['Functional_Molecule', 'Development'],
    'Exists_In_Genotype': [
        lambda t: 'Element' if t in {'Tissue', 'Development_Phase', 'Genotype'} else 'Molecule',
        'Genotype'
    ],
    'Has_Sequence_Identical_To': ['Element1', 'Element2'],
    'Interacts_With': ['Agent', 'Target'],
    'Is_Functionally_Equivalent_To': ['Element1', 'Element2'],
    'Is_Involved_In_Process': ['Participant', 'Process'],
    'Is_Localized_In': [
        lambda t: 'Process' if t in {'Regulatory_Network','Pathway'} else 'Functional_Molecule',
        'Target_Tissue'
    ],
    'Is_Member_Of_Family': ['Element', 'Family'],
    'Is_Protein_Domain_Of': ['Domain', 'Product'],
    'Occurs_During': ['Process', 'Development'],
    'Occurs_In_Genotype': ['Process', 'Genotype'],
    'Regulates_Accumulation': ['Agent', 'Functional_Molecule'],
    'Regulates_Development_Phase': ['Agent', 'Development'],
    'Regulates_Expression': ['Agent', 'DNA'],
    'Regulates_Molecule_Activity': ['Agent', 'Molecule'],
    'Regulates_Process': ['Agent', 'Process'],
    'Regulates_Tissue_Development': ['Agent', 'Target_Tissue'],
    'Transcribes_Or_Translates_To': ['Source', 'Product'],
    'Is_Linked_To': ['Agent1', 'Agent2']
}


def _get_role(role, entity_type):
    if type(role) == str:
        return role
    return role(entity_type)


def write_relations(rels: dict, path: str):
    makedirs(path, exist_ok=True)

    for doc_name in rels.keys():
        i = 1
        with open(join(path, doc_name + '.a2'), 'w', encoding='utf-8') as f:
            for rel in sorted(rels[doc_name], key=lambda x: (x.first_entity.id, x.second_entity.id)):
                rel_schema = _schema.get(rel.type, None)
                first_role, second_role = rel.entities_types

                if rel_schema is not None:
                    first_role = _get_role(rel_schema[0], first_role)
                    second_role = _get_role(rel_schema[1], second_role)

                f.write('R' + str(i) + "\t" + rel.type + ' ' + first_role + ":" + rel.first_entity.id + ' '
                        + second_role + ":" + rel.second_entity.id + '\n')

                i += 1


def main():
    if len(sys.argv) < 4:
        print("Usage: <model-path> <test-path> <out-path> <transformers-props-path>")
        return
    model_path = sys.argv[1]
    docs = load(sys.argv[2])
    out_path = sys.argv[3]
    transformers_props_path = sys.argv[4] if len(sys.argv) > 4 else None

    if transformers_props_path is not None:
        with open(transformers_props_path, 'r', encoding='utf-8') as f, \
                transformer_from_props(json.load(f)) as transformer:
            docs = [transformer.transform(doc) for doc in docs]

    with RelExtClassifier(model_path) as classifier:
        rels = classifier.predict_docs(docs)

    write_relations(rels, out_path)


if __name__ == '__main__':
    main()

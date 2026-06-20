import ir_datasets


def load_dataset():

    # Dataset → CORD-19 TREC-COVID
    dataset = ir_datasets.load(
        "cord19/trec-covid"
    )

    return dataset


def print_dataset_info(dataset):

    print("DATASET")
    print("Name: CORD-19 TREC-COVID")

    print("Docs:", dataset.docs_count())
    print("Queries:", dataset.queries_count())

    try:
        print("Qrels:", dataset.qrels_count())
    except:
        print("Qrels: Not available")
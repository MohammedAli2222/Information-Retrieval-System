import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from services.evaluation_service import EvaluationService

def generate_charts_and_json() -> None:
    cache_file = "cache/evaluation_results_v3.pkl"
    
    if not os.path.exists(cache_file):
        print("Error: Cache file not found. Please run evaluate.py first.")
        return

    print("Loading pre-computed results from cache...")
    with open(cache_file, "rb") as f:
        cached_results = pickle.load(f)

    all_qrels = cached_results["all_qrels"]
    
    results_dict = {
        "TF-IDF": cached_results["tfidf"],
        "BM25": cached_results["bm25"],
        "BERT": cached_results["bert"],
        "Hybrid Serial": cached_results["hs_before"],
        "Hybrid Parallel (Raw Query)": cached_results["hp_before"],
        "Hybrid Parallel (Smart SpellCheck)": cached_results["hp_after_spell"]
    }

    evaluator = EvaluationService()
    final_report = {}

    print("Calculating final metrics and generating condensed report...")
    for model_name, results in results_dict.items():
        metrics = evaluator.evaluate_system(results, all_qrels, k=10)
        
        # شرح الكود باللغة العربية: تخزين المقاييس العامة بشكل مباشر ونظيف
        report_entry = {
            "MAP": round(metrics.get("MAP", 0.0), 4),
            "Recall": round(metrics.get("Recall", 0.0), 4),
            "P@10": round(metrics.get("P@10", 0.0), 4),
            "nDCG@10": round(metrics.get("nDCG", 0.0), 4)
        }
        
        # شرح الكود باللغة العربية: إضافة تقرير الإصابات حصراً للمحرك المتوازي، وضغطه في سطر واحد لكل استعلام
        if model_name == "Hybrid Parallel (Raw Query)":
            hits_report = evaluator.per_query_relevant_retrieval_report(results, all_qrels)
            compact_hits = {}
            for qid, stats in list(hits_report.items())[:50]:
                compact_hits[qid] = f"{int(stats['hit'])}/{int(stats['total_relevant'])} ({stats['recall']:.3f})"
            
            final_report[model_name] = {
                "Metrics": report_entry,
                "Relevant_Hits_Top50": compact_hits
            }
        else:
            final_report[model_name] = report_entry

    os.makedirs("reports", exist_ok=True)
    json_path = "reports/evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    print(f"✅ JSON report saved to: {json_path}")

    # -----------------------------------------
    # رسم المخططات البيانية
    # -----------------------------------------
    models = list(final_report.keys())
    
    # شرح الكود باللغة العربية: استخراج الدقة بناءً على الهيكلة الجديدة (سواء كان الموديل يحوي تقريراً تفصيلياً أم لا)
    p10_scores = [final_report[m]["Metrics"]["P@10"] if "Metrics" in final_report[m] else final_report[m]["P@10"] for m in models]
    map_scores = [final_report[m]["Metrics"]["MAP"] if "Metrics" in final_report[m] else final_report[m]["MAP"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    
    rects1 = ax.bar(x - width/2, p10_scores, width, label='P@10 (Precision)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, map_scores, width, label='MAP', color='#ff7f0e')

    ax.set_ylabel('Scores')
    ax.set_title('Information Retrieval Models Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.legend()

    def autolabel(rects: plt.bar) -> None:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    chart_path = "reports/models_comparison_chart.png"
    plt.savefig(chart_path, dpi=300)
    print(f"✅ Performance chart saved to: {chart_path}")
    
    plt.close()

if __name__ == "__main__":
    generate_charts_and_json()
import pandas as pd
import numpy as np
from app import load_content_model, load_collab_model, recomendation, user_recommendation
from sklearn.model_selection import train_test_split
import csv

# ================================
# Evaluation Metrics
# ================================

def category_precision_at_k(rec_books, test_books, df_all, k):
    test_cats = df_all[df_all['Title'].str.lower().isin(test_books)]['categories'].dropna().str.lower().str.split(', ')
    test_cats = set(cat for sublist in test_cats for cat in sublist)

    rec_cats = df_all[df_all['Title'].str.lower().isin(rec_books)]['categories'].dropna().str.lower().str.split(', ')
    rec_cats = set(cat for sublist in rec_cats for cat in sublist)

    if not test_cats:
        return 0.0
    return len(test_cats & rec_cats) / len(test_cats)

def category_recall_at_k(rec_books, test_books, df_all):
    test_cats = df_all[df_all['Title'].str.lower().isin(test_books)]['categories'].dropna().str.lower().str.split(', ')
    test_cats = set(cat for sublist in test_cats for cat in sublist)

    rec_cats = df_all[df_all['Title'].str.lower().isin(rec_books)]['categories'].dropna().str.lower().str.split(', ')
    rec_cats = set(cat for sublist in rec_cats for cat in sublist)

    if not rec_cats:
        return 0.0
    return len(test_cats & rec_cats) / len(rec_cats)

def category_f1(cat_precision, cat_recall):
    if cat_precision + cat_recall == 0:
        return 0.0
    return 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall)

def category_hit(rec_books, test_books, df_all):
    test_cats = df_all[df_all['Title'].str.lower().isin(test_books)]['categories'].dropna().str.lower().str.split(', ')
    test_cats = set(cat for sublist in test_cats for cat in sublist)

    rec_cats = df_all[df_all['Title'].str.lower().isin(rec_books)]['categories'].dropna().str.lower().str.split(', ')
    rec_cats = set(cat for sublist in rec_cats for cat in sublist)

    return int(bool(test_cats & rec_cats))

def average_precision(recommended, relevant):
    score = 0.0
    hit = 0
    for i, item in enumerate(recommended):
        if item in relevant:
            hit += 1
            score += hit / (i + 1)
    return score / len(relevant) if relevant else 0


# ================================
# Evaluation Function
# ================================

def evaluate_hybrid_model(user_ids, user_books_matrix, df, features, cos_sim, user_sim_df, df_all, k=10):
    precision_scores, recall_scores, f1_scores, hit_scores = [], [], [], []
    failed_users = []

    df['Title'] = df['Title'].str.lower()
    user_books_matrix.columns = [col.lower() for col in user_books_matrix.columns]
    features.index = features.index.str.lower()
    df_all['Title'] = df_all['Title'].str.lower()

    with open('evaluation_log.csv', mode='w', newline='', encoding='utf-8') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(['User_id', 'Test_Books', 'Recommended', 'Category_Precision', 'Category_Recall', 'F1', 'Hit'])

        for user_id in user_ids:
            try:
                user_data = df[df['User_id'] == user_id]
                read_books = user_data['Title'].tolist()

                if len(read_books) < 5:
                    continue

                test_books = [b.lower() for b in read_books[-2:]]
                train_books = [b.lower() for b in set(read_books) - set(test_books)]

                if not train_books:
                    failed_users.append((user_id, 'no train books'))
                    continue

                # Content-based from multiple books
                content_recs = []
                for book in train_books[:3]:
                    if book in features.index:
                        recs = recomendation(book, features, cos_sim, top_n=15)['Recommended Books'].str.lower().tolist()
                        content_recs.extend(recs)
                content_recs = list(pd.Series(content_recs).drop_duplicates())

                # Collaborative filtering
                collab_recs = user_recommendation(user_id, user_sim_df, user_books_matrix, df, top_n=30)['Recommended Books'].str.lower().tolist()

                # Hybrid combination
                hybrid = list(pd.concat([pd.Series(collab_recs), pd.Series(content_recs)]).drop_duplicates())[:k]

                cat_precision = category_precision_at_k(hybrid, test_books, df_all, k)
                cat_recall = category_recall_at_k(hybrid, test_books, df_all)
                cat_f1 = category_f1(cat_precision, cat_recall)
                cat_hit = category_hit(hybrid, test_books, df_all)

                precision_scores.append(cat_precision)
                recall_scores.append(cat_recall)
                f1_scores.append(cat_f1)
                hit_scores.append(cat_hit)

                writer.writerow([user_id, "; ".join(test_books), "; ".join(hybrid), cat_precision, cat_recall, cat_f1, cat_hit])
                # print(f"User {user_id} | Test: {test_books} | Cat Precision: {cat_precision:.4f} | Recall: {cat_recall:.4f} | F1: {cat_f1:.4f} | Hit: {cat_hit}")

            except Exception as e:
                failed_users.append((user_id, str(e)))
                continue

    print(f"Evaluated {len(precision_scores)} users")
    print(f"Average Category Precision@{k}: {np.mean(precision_scores):.4f}")
    print(f"Average Category Recall@{k}: {np.mean(recall_scores):.4f}")
    print(f"Average Category F1@{k}: {np.mean(f1_scores):.4f}")
    print(f"Hit Rate: {np.mean(hit_scores):.4f}")

    if failed_users:
        with open('evaluation_failed_users.csv', mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['User_id', 'Reason'])
            writer.writerows(failed_users)

# ================================
# Run Evaluation
# ================================

if __name__ == '__main__':
    features, cos_sim = load_content_model()
    df, user_books_matrix, user_sim_df = load_collab_model()
    df_all = pd.read_csv('dataset/use/content_df.csv')

    unique_users = df['User_id'].unique()
    sampled_users = np.random.choice(unique_users, size=100, replace=False)

    evaluate_hybrid_model(sampled_users, user_books_matrix, df, features, cos_sim, user_sim_df, df_all, k=10)
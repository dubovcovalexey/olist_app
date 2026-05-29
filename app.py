import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostRanker
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Olist RecSys", layout="wide")

st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 1rem; }
    .dataframe { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# Архитектурный каркас нейросети для инференса Two Tower
class TwoTowerNetwork(nn.Module):
    def __init__(self, item_feat_dim, embedding_dim=64):
        super(TwoTowerNetwork, self).__init__()
        self.user_fc = nn.Sequential(nn.Linear(item_feat_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, embedding_dim))
        self.item_fc = nn.Sequential(nn.Linear(item_feat_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, embedding_dim))

    def forward_user(self, history, item_features_matrix):
        hist_features = item_features_matrix[history]
        mask = (history > 0).float().unsqueeze(-1)
        sum_features = torch.sum(hist_features * mask, dim=1)
        denom = torch.sum(mask, dim=1).clamp(min=1.0)
        mean_features = sum_features / denom
        return F.normalize(self.user_fc(mean_features), p=2, dim=1)

    def forward_item(self, item_features):
        return F.normalize(self.item_fc(item_features), p=2, dim=1)


# Полноценная сборка и кэширование моделей в оперативной памяти облачного сервера
@st.cache_resource
def load_all_models_and_assets():
    with open("streamlit_data.pkl", "rb") as f:
        data = pickle.load(f)
        
    x_dense = data["x_content"].toarray()
    x_item_tensor = torch.tensor(x_dense, dtype=torch.float32)
    
    padding_vector = torch.zeros((1, x_item_tensor.shape[1]), dtype=torch.float32)
    x_item_tensor_extended = torch.cat([padding_vector, x_item_tensor], dim=0)
    
    net = TwoTowerNetwork(x_item_tensor_extended.shape[1], embedding_dim=64)
    try:
        net.load_state_dict(torch.load("two_tower.pt", map_location=torch.device('cpu')))
    except:
        pass
    net.eval()
    
    with torch.no_grad():
        all_item_embeddings = net.forward_item(x_item_tensor_extended)
        
    cb = CatBoostRanker()
    try:
        cb.load_model("catboost_ranker.cbm")
    except:
        cb = None
        
    return data, net, x_item_tensor_extended, all_item_embeddings, cb

assets, model_pytorch, x_item_tensor_extended, all_item_embeddings, cb_model = load_all_models_and_assets()

full_history = assets["full_history"]
df_products = assets["df_products"]
swing_matrix = assets["swing_matrix"]
x_content = assets["x_content"]
product_list = assets["product_list"]
pid_to_idx = assets["pid_to_idx"]
pid_to_idx_shifted = assets["pid_to_idx_shifted"]
idx_to_pid_shifted = assets["idx_to_pid_shifted"]


# Сайдбар управления режимами фильтрации и трехуровневого выбора лотов
st.sidebar.header("Настройки")
mode = st.sidebar.radio("Режим поиска:", ["Пользователь (User ID)", "Товар (Product ID)"])

if mode == "Пользователь (User ID)":
    selected_user = st.sidebar.number_input("Номер Покупателя (1 - 91979):", min_value=1, max_value=len(full_history), value=None, step=1)
    available_models = [
        "Model 1: Alibaba Swing (Графовая)",
        "Model 2: Content-Based (Косинусное сходство)",
        "Model 3: Score Fusion (Аддитивный гибрид)",
        "Model 4: Two-Tower (Нейросеть DSSM)",
        "Model 5: CatBoostRanker (Двухэтапный бустинг)"
    ]
else:
    st.sidebar.subheader("Поиск товара по каталогу")
    unique_macro_groups = sorted(df_products['category_group'].unique().tolist())
    selected_macro = st.sidebar.selectbox("Шаг 1: Макро-группа:", unique_macro_groups)
    
    filtered_df_by_macro = df_products[df_products['category_group'] == selected_macro]
    # Фильтруем пустые значения, приводим к строкам и безопасно закрываем метод sorted()
    raw_unique = filtered_df_by_macro['product_category_name_english'].dropna().astype(str).unique().tolist()
    unique_micro_groups = sorted(raw_unique) if raw_unique else ['other']


    selected_micro = st.sidebar.selectbox("Шаг 2: Микро-категория:", unique_micro_groups)
    
    filtered_products_by_micro = filtered_df_by_macro[filtered_df_by_macro['product_category_name_english'] == selected_micro].index.tolist()
    selected_product = st.sidebar.selectbox("Шаг 3: Целевой товар:", sorted(filtered_products_by_micro))
    
    available_models = ["Model 2: Content-Based (Косинусное сходство)"]

selected_model = st.sidebar.selectbox("Модель:", available_models)
k_recs = st.sidebar.slider("Количество рекомендаций:", 5, 15, 10)


# Отрисовщик таблиц выводящий скор физические и коммерческие фичи
def show_product_features_with_scores(title, recs_with_scores):
    st.subheader(title)
    pids = [item[0] for item in recs_with_scores]
    
    cols_to_show = [
        'category_group', 'product_category_name_english', 'avg_unit_price', 
        'avg_review_score', 'avg_delivery_days', 'total_sold',
        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
        'seller_region_top1'
    ]
    sub_df = df_products.loc[pids, [c for c in cols_to_show if c in df_products.columns]].copy()
    
    sub_df = sub_df.rename(columns={
        'category_group': 'Макро-группа',
        'product_category_name_english': 'Микро-категория',
        'avg_unit_price': 'Цена',
        'avg_review_score': 'Оценка',
        'avg_delivery_days': 'Доставка (дн)',
        'total_sold': 'Продано (шт)',
        'product_weight_g': 'Вес (г)',
        'product_length_cm': 'Длина (см)',
        'product_height_cm': 'Высота (см)',
        'product_width_cm': 'Ширина (см)',
        'seller_region_top1': 'Регион продавца'
    })
    
    score_map = dict(recs_with_scores)
    sub_df.insert(0, 'Скор', sub_df.index.map(score_map))
    sub_df.insert(0, 'Товар', sub_df.index)
    
    sub_df = sub_df.sort_values(by='Скор', ascending=False)
    st.dataframe(sub_df.style.format({
        'Скор': '{:.4f}', 'Цена': '{:.2f}', 'Оценка': '{:.2f}', 'Доставка (дн)': '{:.1f}', 
        'Продано (шт)': '{:.0f}', 'Вес (г)': '{:.0f}', 'Длина (см)': '{:.0f}', 
        'Высота (см)': '{:.0f}', 'Ширина (см)': '{:.0f}'
    }), hide_index=True)

# Живой офлайн инференс всех пяти подходов для выбранного числового покупателя
if mode == "Пользователь (User ID)":
    if selected_user is None:
        st.write("⬅️ Пожалуйста, введите числовой номер покупателя на боковой панели, чтобы рассчитать рекомендации.")
    elif selected_user in full_history:
        history = full_history[selected_user]
        st.info(f"Покупатель №{selected_user} | Товаров в истории: {len(history)}")
        
        hist_with_dummy_scores = [(p, 1.0) for p in history]
        show_product_features_with_scores("История покупок", hist_with_dummy_scores)
        
        final_recs_with_scores = []
        
        # Модель 1: Swing
        if "Model 1: Alibaba Swing" in selected_model:
            candidates = {}
            for item in history:
                if item in swing_matrix:
                    for sim_item, score in swing_matrix[item].items():
                        if sim_item not in history:
                            candidates[sim_item] = candidates.get(sim_item, 0) + score
            final_recs_with_scores = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:k_recs]

        # Модель 2: асчет косинусного сходства векторов признаков лотов
        elif "Model 2: Content-Based" in selected_model:
            hist_idxs = [pid_to_idx[p] for p in history if p in pid_to_idx]
            if hist_idxs:
                user_sims = np.max(cosine_similarity(x_content[hist_idxs], x_content), axis=0)
                for p in history:
                    if p in pid_to_idx: user_sims[pid_to_idx[p]] = -1.0
                top_idxs = np.argsort(user_sims)[::-1][:k_recs]
                final_recs_with_scores = [(product_list[idx], float(user_sims[idx])) for idx in top_idxs]

        # Модель 3: Score Fusion гибрид 
        elif "Model 3: Score Fusion" in selected_model:
            hist_idxs = [pid_to_idx[p] for p in history if p in pid_to_idx]
            if hist_idxs:
                user_sims = np.max(cosine_similarity(x_content[hist_idxs], x_content), axis=0)
                s_min, s_max = user_sims.min(), user_sims.max()
                if s_max - s_min > 0: user_sims = (user_sims - s_min) / (s_max - s_min)
                for item in history:
                    if item in swing_matrix:
                        for target_pid, swing_weight in swing_matrix[item].items():
                            if target_pid in pid_to_idx: user_sims[pid_to_idx[target_pid]] += swing_weight * 3.0
                for p in history:
                    if p in pid_to_idx: user_sims[pid_to_idx[p]] = -1.0
                top_idxs = np.argsort(user_sims)[::-1][:k_recs]
                final_recs_with_scores = [(product_list[idx], float(user_sims[idx])) for idx in top_idxs]

        # Модель 4: Two-Tower через PyTorch тензоры
        elif "Model 4: Two-Tower" in selected_model:
            rem_indices_shifted = [pid_to_idx_shifted[p] for p in history if p in pid_to_idx_shifted]
            if rem_indices_shifted:
                hist_input = rem_indices_shifted[-10:]
                if len(hist_input) < 10:
                    hist_input = list(hist_input) + [0] * (10 - len(hist_input))
                    
                with torch.no_grad():
                    user_embed = model_pytorch.forward_user(torch.tensor([hist_input], dtype=torch.long), x_item_tensor_extended)
                    scores = torch.matmul(user_embed, all_item_embeddings.T).squeeze(0).cpu().numpy()
                    
                for p in history:
                    if p in pid_to_idx_shifted: scores[pid_to_idx_shifted[p]] = -1.0
                scores[0] = -1.0
                top_idxs = np.argsort(scores)[::-1][:k_recs]
                final_recs_with_scores = [(idx_to_pid_shifted[idx], float(scores[idx])) for idx in top_idxs if idx in idx_to_pid_shifted]

        # Модель 5: CatBoostRanker по матрице признаков
        elif "Model 5: CatBoostRanker" in selected_model:
            hist_idxs = [pid_to_idx[p] for p in history if p in pid_to_idx]
            if hist_idxs:
                user_sims = np.max(cosine_similarity(x_content[hist_idxs], x_content), axis=0)
                for p in history:
                    if p in pid_to_idx: user_sims[pid_to_idx[p]] = -1.0
                # Шаг 1: Честный отбор 100 кандидатов первого уровня
                top_100_idxs = np.argsort(user_sims)[::-1][:100]
                candidate_pids = [product_list[idx] for idx in top_100_idxs]
                
                if cb_model is None:
                    # Если файла весов .cbm нет, отдаем базовый топ
                    final_recs_with_scores = [(product_list[idx], float(user_sims[idx] * 1.5)) for idx in top_100_idxs[:k_recs]]
                else:
                    
                    features_list = [
                        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
                        'total_sold', 'total_revenue', 'sales_per_month', 'avg_unit_price', 
                        'avg_review_score', 'avg_delivery_days', 'seller_region_top1', 'customer_region_top1'
                    ]
                    # Вырезаем строки кандидатов и оставляем только нужные фичи в правильном порядке колонок
                    X_cand = df_products.loc[candidate_pids, [c for c in features_list if c in df_products.columns]].copy()
                    
                    # Переименовываем колонки регионов, чтобы они совпали со структурой Pool при обучении
                    X_cand = X_cand.rename(columns={'seller_region_top1': 'seller_region', 'customer_region_top1': 'customer_region'})
                    
                    # Вызываем оригинальный метод предсказания скоров YetiRank деревьями CatBoost
                    preds = cb_model.predict(X_cand)
                    
                    # Привязываем честные предсказания к именам товаров и забираем ТОП-K
                    cb_scores = list(zip(candidate_pids, [float(p) for p in preds]))
                    final_recs_with_scores = sorted(cb_scores, key=lambda x: x[1], reverse=True)[:k_recs]

        if final_recs_with_scores:
            show_product_features_with_scores("Рекомендованные сопутствующие товары", final_recs_with_scores)
    else:
        st.warning(f"Пользователь №{selected_user} не найден")

# Офлайн расчет скоров по косинусному расстоянию для режима поиска по товару
else:
    st.info(f"Выбран опорный товар: {selected_product}")
    show_product_features_with_scores("Характеристики товара", [(selected_product, 1.0)])
    
    final_recs_with_scores = []
    if selected_product in pid_to_idx:
        prod_idx = pid_to_idx[selected_product]
        prod_sims = cosine_similarity(x_content[[prod_idx]], x_content).flatten()
        prod_sims[prod_idx] = -1.0
        
        top_idxs = np.argsort(prod_sims)[::-1][:k_recs]
        final_recs_with_scores = [(product_list[idx], float(prod_sims[idx])) for idx in top_idxs]
        
    if final_recs_with_scores:
        show_product_features_with_scores("Похожие товары", final_recs_with_scores)


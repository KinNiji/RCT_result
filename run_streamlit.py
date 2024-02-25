import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


question = pd.read_excel('data/explainations.xlsx', sheet_name='question')
scale = pd.read_excel('data/explainations.xlsx', sheet_name='scale')
variables = pd.read_excel('data/explainations.xlsx', sheet_name='variables').set_index('feature')['explanation'].to_dict()
df = pd.read_csv('result/top10_with_explanation.csv')

def mapping_variable(variable):
    explanation = variables.get(variable, None)
    if 'interaction' in variable:
        return f'{explanation}(交互效应)'
    elif 'zubie' in variable:
        return f'{explanation}(组别效应)'
    return explanation

def mapping_symptom(variable):
    # 找到name_EN=variable的行
    row = question[question['name_EN'] == variable]
    scale_id = row['scale_id'].values[0]
    scale_name = scale[scale['id'] == scale_id]['name_CN'].values[0]
    name = row['name_CN'].values[0]
    _type = row['type'].values[0]
    text = row['text'].values[0]
    options = row['options'].values[0]
    values = row['values'].values[0]
    if _type != '统计项':
        if _type != '填空':
            return f'量表：{scale_name}, 问题：{text}, 选项：{options}, 分数：{values}'
        else:
            return f'量表：{scale_name}, 问题：{text}'
    else:
        return f'量表：{scale_name}, 描述：{text}'

def mapping_scale_name(variable):
    return scale[scale['id'] == question[question['name_EN'] == variable]['scale_id'].values[0]]['name_CN'].values[0]

def mapping_question_text(variable):
    return question[question['name_EN'] == variable]['text'].values[0]

st.title('Select a row to view details')

index = st.selectbox('Row Index', df.index)

if index < 0 or index >= len(df):
    st.error('Please select an index')
else:
    row = df.iloc[index]
    statistics = {
        'real_before_vs_after_p': row['p_value_1'],
        'real_before_vs_after_t': row['t_statistic_1'],
        'sham_before_vs_after_p': row['p_value_2'],
        'sham_before_vs_after_t': row['t_statistic_2'],
        'real_before_vs_sham_before_p': row['p_value_3'],
        'real_before_vs_sham_before_t': row['t_statistic_3'],
        'real_after_vs_sham_after_p': row['p_value_4'],
        'real_after_vs_sham_after_t': row['t_statistic_4'],
    }

    st.markdown('#### 真刺激组治疗前后差异\tp-value: {:.4f}\tt-statistic: {:.2f}'.format(statistics['real_before_vs_after_p'], statistics['real_before_vs_after_t']))
    st.markdown('#### 伪刺激组治疗前后差异\tp-value: {:.4f}\tt-statistic: {:.2f}'.format(statistics['sham_before_vs_after_p'], statistics['sham_before_vs_after_t']))
    st.markdown('#### 真伪刺激组治疗前差异\tp-value: {:.4f}\tt-statistic: {:.2f}'.format(statistics['real_before_vs_sham_before_p'], statistics['real_before_vs_sham_before_t']))
    st.markdown('#### 真伪刺激组治疗后差异\tp-value: {:.4f}\tt-statistic: {:.2f}'.format(statistics['real_after_vs_sham_after_p'], statistics['real_after_vs_sham_after_t']))

    symptom_variables = row['Variables'][1: -1].replace('\'', '').split(', ')
    mean_b_t0_real = row['mean_b_t0_real'][1: -1].replace('\'', '').split(', ')
    mean_b_t0_sham = row['mean_b_t0_sham'][1: -1].replace('\'', '').split(', ')
    mean_b_t2_real = row['mean_b_t2_real'][1: -1].replace('\'', '').split(', ')
    mean_b_t2_sham = row['mean_b_t2_sham'][1: -1].replace('\'', '').split(', ')
    mean_b_t0_real = [round(float(x), 2) for x in mean_b_t0_real]
    mean_b_t0_sham = [round(float(x), 2) for x in mean_b_t0_sham]
    mean_b_t2_real = [round(float(x), 2) for x in mean_b_t2_real]
    mean_b_t2_sham = [round(float(x), 2) for x in mean_b_t2_sham]
    imaging_variables = row['imaging_variables_pearson'][1: -1].replace('\'', '').split(', ')
    imaging_correlations = row['imaging_correlations_pearson'][1: -1].split(', ')
    imaging_correlations = [round(float(x), 3) for x in imaging_correlations]
    imaging_p_values = row['imaging_p_value_pearson'][1: -1].split(', ')
    imaging_p_values = [float(x) for x in imaging_p_values]

    voice_variables = row['voice_variables_pearson'][1: -1].replace('\'', '').split(', ')
    voice_correlations = row['voice_correlations_pearson'][1: -1].split(', ')
    voice_correlations = [round(float(x), 3) for x in voice_correlations]
    voice_p_values = row['voice_p_value_pearson'][1: -1].split(', ')
    voice_p_values = [float(x) for x in voice_p_values]
    
    symptom_df = pd.DataFrame({
        'symptom_variables': symptom_variables, 
        'real_before_mean': mean_b_t0_real,
        'sham_before_mean': mean_b_t0_sham,
        'real_after_mean': mean_b_t2_real,
        'sham_after_mean': mean_b_t2_sham
    })

    st.title('症状组合')

    st.markdown('将原始数据分为4组（真刺激治疗前、真刺激治疗后、伪刺激治疗前、伪刺激治疗后），将组合的所有特征值按样本相加，分别对真刺激组治疗前后差异、伪刺激组治疗前后差异、真伪刺激组治疗前差异、真伪刺激组治疗后差异进行t检验，挑选出:red[真伪刺激组在治疗前无明显差异]，:red[真刺激组治疗前后有明显差异]，且:red[治疗前后真刺激组差异大于假刺激组差异]的特征组合并记录。')
    st.markdown('之后将所有满足条件的特征组合，按治疗前后差异最大的进行排列，和按真刺激组差异大于伪刺激组差异进行排列，分别取top k拼接在一起并去重。')

    st.write(symptom_df)

    symptom_df['scale_name'] = symptom_df['symptom_variables'].apply(mapping_scale_name)
    symptom_df['text'] = symptom_df['symptom_variables'].apply(mapping_question_text)

    st.write(symptom_df[['symptom_variables', 'scale_name', 'text']])
    
    real_symptom_before_df = pd.DataFrame({
        'symptom_variables': symptom_variables, 
        't': 'T0',
        'mean': mean_b_t0_real
    })
    real_symptom_after_df = pd.DataFrame({
        'symptom_variables': symptom_variables, 
        't': 'T2',
        'mean': mean_b_t2_real
    })
    real_symptom_df = pd.concat([real_symptom_before_df, real_symptom_after_df])
    
    sham_symptom_before_df = pd.DataFrame({
        'symptom_variables': symptom_variables, 
        't': 'T0',
        'mean': mean_b_t0_sham
    })
    sham_symptom_after_df = pd.DataFrame({
        'symptom_variables': symptom_variables, 
        't': 'T2',
        'mean': mean_b_t2_sham
    })
    sham_symptom_df = pd.concat([sham_symptom_before_df, sham_symptom_after_df])

    col1, col2 = st.columns(2)
    col1.markdown('#### 真刺激组')
    col1.line_chart(data=real_symptom_df, x='t', y='mean', color='symptom_variables')
    col2.markdown('#### 伪刺激组')
    col2.line_chart(data=sham_symptom_df, x='t', y='mean', color='symptom_variables')


    fig, (ax_real, ax_sham) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    time_points = [0, 1]  # 0 for 'Before Treatment', 1 for 'After Treatment'
    for i, variable in enumerate(symptom_variables):
        ax_real.plot(time_points, [mean_b_t0_real[i], mean_b_t2_real[i]], 'o-', label=variable)
    for i, variable in enumerate(symptom_variables):
        ax_sham.plot(time_points, [mean_b_t0_sham[i], mean_b_t2_sham[i]], 'o-', label=variable)
    ax_real.set_xticks(time_points)
    ax_real.set_xticklabels(['Before', 'After'])
    ax_sham.set_xticks(time_points)
    ax_sham.set_xticklabels(['Before', 'After'])
    ax_real.set_title('Real')
    ax_sham.set_title('Sham')
    ax_sham.legend(title='Symptom Variables', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Leave space for the whole figure title and legend

    st.pyplot(fig)

    imaging_df = pd.DataFrame({
        'imaging_variables': imaging_variables, 
        'imaging_correlations': imaging_correlations,
        'imaging_p_values': imaging_p_values
    })
    imaging_df['imaging_variables'] = imaging_df['imaging_variables'].apply(mapping_variable)
    voice_df = pd.DataFrame({
        'voice_variables': voice_variables, 
        'voice_correlations': voice_correlations,
        'voice_p_values': voice_p_values
    })
    voice_df['voice_variables'] = voice_df['voice_variables'].apply(mapping_variable)
    imaging_and_voice_df = pd.concat([imaging_df, voice_df], axis=1)

    st.title('相关度高的影像和声音特征')

    st.markdown('对top 2k的每一个特征组合，其中每个特征与所有影像与声音特征计算皮尔逊相关系数，取出相关度最高的n个影像和声音特征（n=特征组合中特征个数）。')

    st.write(imaging_and_voice_df)

    st.line_chart(data=imaging_df, x='imaging_variables', y='imaging_correlations')

    st.line_chart(data=voice_df, x='voice_variables', y='voice_correlations')

    st.title('GPT解释')

    st.markdown(row['explanation'])




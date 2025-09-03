import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import h5py
import scipy.io

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results", type=str, nargs='+', required=True, help="Path(s) to results file(s) (.pkl, .json, or .mat)")
    parser.add_argument("--labels", type=str, nargs='+', help="Labels for each result file in the plots (must match number of result files)")
    parser.add_argument("--plot_type", type=str, choices=["velocity", "snr", "step", "all"], default="all",
                        help="Type of plot to generate (velocity, snr, step, or all)")
    parser.add_argument("--metric", type=str, choices=["nmse", "se", "both"], default="both",
                        help="Metric to plot (nmse, se, or both)")
    parser.add_argument("--velocity", type=int, default=30,
                        help="Fixed velocity value for SNR plots (km/h)")
    parser.add_argument("--snr", type=float, default=10.0,
                        help="Fixed SNR value for velocity plots (dB)")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--se_type", type=str, choices=["estimated", "true", "both"], default="both",
                        help="Type of SE to plot (when metric=se)")
    parser.add_argument("--save_format", type=str, default="png",
                        help="Figure save format")
    parser.add_argument("--frame_interval", type=float, default=0.625,
                        help="Frame interval in milliseconds")
    parser.add_argument("--save_matlab", action="store_true", default=True,
                        help="Save plot data as MATLAB file for further editing")
    return parser.parse_args()

def load_results(results_path):
    """Load results from file based on extension"""
    ext = os.path.splitext(results_path)[1].lower()
    
    if ext == '.pkl':
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
            return data['results']
    
    elif ext == '.json':
        with open(results_path, 'r') as f:
            data = json.load(f)
            
            # Convert JSON structure back to our internal format
            results = {
                'nmse': {'avg': {}, 'per_step': {}},
                'se': {'est': {'avg': {}, 'per_step': {}},
                      'true': {'avg': {}, 'per_step': {}}}
            }
            
            # Parse NMSE results
            for key, value in data['nmse']['average'].items():
                parts = key.split('_')
                v = int(parts[0].replace('km/h', ''))
                s = float(parts[1].replace('dB', ''))
                results['nmse']['avg'][(v, s)] = value
            
            for key, values in data['nmse']['per_step'].items():
                parts = key.split('_')
                v = int(parts[0].replace('km/h', ''))
                s = float(parts[1].replace('dB', ''))
                results['nmse']['per_step'][(v, s)] = values
            
            # Parse SE results
            for key, value in data['se']['estimated']['average'].items():
                parts = key.split('_')
                v = int(parts[0].replace('km/h', ''))
                s = float(parts[1].replace('dB', ''))
                results['se']['est']['avg'][(v, s)] = value
            
            for key, values in data['se']['estimated']['per_step'].items():
                parts = key.split('_')
                v = int(parts[0].replace('km/h', ''))
                s = float(parts[1].replace('dB', ''))
                results['se']['est']['per_step'][(v, s)] = values
            
            for key, value in data['se']['true']['average'].items():
                parts = key.split('_')
                v = int(parts[0].replace('km/h', ''))
                s = float(parts[1].replace('dB', ''))
                results['se']['true']['avg'][(v, s)] = value
                
            for key, values in data['se']['true']['per_step'].items():
                parts = key.split('_')
                v = int(parts[0].replace('km/h', ''))
                s = float(parts[1].replace('dB', ''))
                results['se']['true']['per_step'][(v, s)] = values
            
            return results
    
    elif ext == '.mat':
        mat_data = scipy.io.loadmat(results_path)
        
        results = {
            'nmse': {'avg': {}, 'per_step': {}},
            'se': {'est': {'avg': {}, 'per_step': {}},
                  'true': {'avg': {}, 'per_step': {}}}
        }
        
        # Extract matrices
        vel_values = mat_data['velocity_values'].flatten()
        snr_values = mat_data['snr_values'].flatten()
        
        # Convert matrices to our format
        nmse_matrix = mat_data['nmse_matrix']
        for i, v in enumerate(vel_values):
            for j, s in enumerate(snr_values):
                if not np.isnan(nmse_matrix[i, j]):
                    results['nmse']['avg'][(int(v), float(s))] = float(nmse_matrix[i, j])
        
        se_est_matrix = mat_data['se_est_matrix']
        for i, v in enumerate(vel_values):
            for j, s in enumerate(snr_values):
                if not np.isnan(se_est_matrix[i, j]):
                    results['se']['est']['avg'][(int(v), float(s))] = float(se_est_matrix[i, j])
        
        se_true_matrix = mat_data['se_true_matrix']
        for i, v in enumerate(vel_values):
            for j, s in enumerate(snr_values):
                if not np.isnan(se_true_matrix[i, j]):
                    results['se']['true']['avg'][(int(v), float(s))] = float(se_true_matrix[i, j])
        
        return results
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def nmse_to_db(nmse_values):
    """Convert NMSE values to dB scale"""
    return 10 * np.log10(nmse_values)

def save_matlab_data(data_dict, output_path):
    """Save plot data to MATLAB format for further editing"""
    # 确保文件名不包含小数点，将其替换为下划线
    output_path = output_path.replace('.0.mat', '_0.mat')
    
    # 将labels转换为适合MATLAB的格式
    if 'labels' in data_dict:
        # 创建一个特殊结构，使scipy.io.savemat将其保存为cell数组
        labels_cell = np.array(data_dict['labels'], dtype=object)
        data_dict['labels'] = labels_cell
    
    # 确保标题和标签是字符串而不是数组
    if 'title_text' in data_dict:
        # 转换为字符串以避免MATLAB中的问题
        title_str = str(data_dict['title_text'])
        data_dict['title_text'] = title_str
    
    if 'x_label' in data_dict:
        data_dict['x_label'] = str(data_dict['x_label'])
    
    if 'y_label' in data_dict:
        data_dict['y_label'] = str(data_dict['y_label'])
    
    # Create a dictionary with all plot data
    scipy.io.savemat(output_path, data_dict)
    
    # Create a simple MATLAB script to recreate the figure
    script_path = output_path.replace('.mat', '_script.m')
    mat_filename = os.path.basename(output_path)
    
    with open(script_path, 'w') as f:
        f.write("% MATLAB脚本用于创建和自定义图表\n")
        f.write("% 由Python生成，支持进一步自定义\n\n")
        
        # 添加可配置参数部分
        f.write("%% 可配置参数 - 修改这些参数以自定义图表 %%\n\n")
        
        # 图表尺寸和字体
        f.write("% 图表尺寸和字体\n")
        f.write("figWidth = 800;              % 图表宽度（像素）\n")
        f.write("figHeight = 600;             % 图表高度（像素）\n")
        f.write("axesFontSize = 12;           % 坐标轴字体大小\n")
        f.write("titleFontSize = 16;          % 标题字体大小\n")
        f.write("labelFontSize = 14;          % 坐标轴标签字体大小\n")
        f.write("legendFontSize = 12;         % 图例字体大小\n")
        f.write("legendLocation = 'best';     % 图例位置: 'best', 'northeast', 'northwest', 等\n\n")
        
        # 线条样式和标记
        f.write("% 线条样式和标记\n")
        f.write("lineWidth = 2;               % 线条宽度\n")
        f.write("markerSize = 8;              % 标记大小\n")
        f.write("perfectCSILineStyle = '--';  % Perfect CSI线型: '-', '--', ':', '-.'\n")
        f.write("modelLineStyle = '-';        % 模型线型: '-', '--', ':', '-.'\n\n")
        
        # 定义不同的标记类型
        f.write("% 标记类型 - 为每个模型定义不同的标记\n")
        f.write("markerStyles = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h', '*'};\n\n")
        
        # 定义颜色方案
        f.write("% 颜色方案 - 手动定义比默认颜色更鲜明的颜色\n")
        f.write("% 可以修改这些颜色以匹配您的喜好\n")
        f.write("customColors = [\n")
        f.write("    0.0000, 0.4470, 0.7410;  % 蓝色\n")
        f.write("    0.8500, 0.3250, 0.0980;  % 橙色\n")
        f.write("    0.4660, 0.6740, 0.1880;  % 绿色\n")
        f.write("    0.9350, 0.1780, 0.2840;  % 红色\n")
        f.write("    0.4940, 0.1840, 0.5560;  % 紫色\n")
        f.write("    0.3010, 0.7450, 0.9330;  % 青色\n")
        f.write("    0.6350, 0.0780, 0.1840;  % 棕红色\n")
        f.write("    0.0000, 0.0000, 0.0000;  % 黑色\n")
        f.write("    0.7500, 0.7500, 0.0000;  % 黄色\n")
        f.write("    0.2500, 0.2500, 0.2500;  % 灰色\n")
        f.write("];\n\n")
        
        # Perfect CSI特殊设置
        f.write("% Perfect CSI特殊设置\n")
        f.write("perfectCSIColor = [0, 0, 0];  % 黑色\n")
        f.write("perfectCSIMarker = 's';       % 方形标记\n")
        f.write("perfectCSILineWidth = 2;      % 线宽\n\n")
        
        # 网格和背景设置
        f.write("% 网格和背景设置\n")
        f.write("gridAlpha = 0.3;             % 网格线透明度\n")
        f.write("gridStyle = '--';            % 网格线样式\n")
        f.write("backgroundColor = 'white';   % 背景颜色\n\n")
        
        # 图表导出设置
        f.write("% 图表导出设置\n")
        f.write("saveFigure = true;           % 是否保存图表\n")
        f.write("outputFormat = 'fig';        % 输出格式: 'fig', 'png', 'eps', 'pdf'\n")
        f.write("figureResolution = 300;      % 解析度 (dpi)\n\n")
        
        f.write("%% 加载数据 %%\n")
        f.write(f"load('{mat_filename}');\n\n")
        
        f.write("%% 创建图表 %%\n")
        f.write("fig = figure('Position', [100, 100, figWidth, figHeight], 'Color', backgroundColor);\n")
        f.write("hold on;\n\n")
        
        # 输出变量信息以帮助调试
        f.write("% 调试信息\n")
        f.write("disp('变量信息:');\n")
        f.write("who\n\n")
        
        # 用于跟踪所有绘图句柄和图例项
        f.write("% 跟踪图例条目\n")
        f.write("plot_handles = [];\n")
        f.write("legend_entries = {};\n\n")
        
        # 修改脚本以处理不同类型的labels变量
        f.write("%% 绘制数据 %%\n")
        f.write("if exist('labels', 'var') && exist('x_values', 'var')\n")
        f.write("    % 确保labels是cell数组\n")
        f.write("    if ~iscell(labels)\n")
        f.write("        labels_cell = cell(length(labels), 1);\n")
        f.write("        for i = 1:length(labels)\n")
        f.write("            if ischar(labels(i))\n")
        f.write("                labels_cell{i} = labels(i,:);\n")
        f.write("            else\n")
        f.write("                labels_cell{i} = ['Line ', num2str(i)];\n")
        f.write("            end\n")
        f.write("        end\n")
        f.write("        labels = labels_cell;\n")
        f.write("    end\n\n")
        
        f.write("    % 打印标签以进行调试\n")
        f.write("    disp('标签内容:');\n")
        f.write("    for i = 1:length(labels)\n")
        f.write("        disp(labels{i});\n")
        f.write("    end\n\n")
        
        f.write("    % 绘制每条线\n")
        f.write("    for i = 0:length(labels)-1\n")
        f.write("        varname = ['y_values_', num2str(i)];\n")
        f.write("        if exist(varname, 'var')\n")
        f.write("            data = eval(varname);\n")
        f.write("            if length(data) <= length(x_values)\n")
        f.write("                % 检查是否是Perfect CSI线\n")
        f.write("                is_perfect_csi = false;\n")
        f.write("                if iscell(labels) && i+1 <= length(labels)\n")
        f.write("                    label_text = labels{i+1};\n")
        f.write("                    if ischar(label_text) && strcmp(label_text, 'Perfect CSI')\n")
        f.write("                        is_perfect_csi = true;\n")
        f.write("                    end\n")
        f.write("                end\n\n")
        
        f.write("                if is_perfect_csi\n")
        f.write("                    h = plot(x_values(1:length(data)), data, ...\n")
        f.write("                        [perfectCSILineStyle, perfectCSIMarker], ...\n")
        f.write("                        'LineWidth', perfectCSILineWidth, ...\n")
        f.write("                        'MarkerSize', markerSize, ...\n")
        f.write("                        'Color', perfectCSIColor);\n")
        f.write("                    plot_handles = [plot_handles, h];\n")
        f.write("                    legend_entries{end+1} = 'Perfect CSI';\n")
        f.write("                else\n")
        f.write("                    % 计算颜色和标记索引（确保在界限内）\n")
        f.write("                    colorIdx = mod(i, size(customColors, 1)) + 1;\n")
        f.write("                    markerIdx = mod(i, length(markerStyles)) + 1;\n")
        f.write("                    \n")
        f.write("                    % 获取标签文本\n")
        f.write("                    if iscell(labels) && i+1 <= length(labels)\n")
        f.write("                        label_text = labels{i+1};\n")
        f.write("                    else\n")
        f.write("                        label_text = ['Line ', num2str(i+1)];\n")
        f.write("                    end\n")
        f.write("                    \n")
        f.write("                    % 绘制带有自定义样式的线\n")
        f.write("                    h = plot(x_values(1:length(data)), data, ...\n")
        f.write("                        [modelLineStyle, markerStyles{markerIdx}], ...\n")
        f.write("                        'LineWidth', lineWidth, ...\n")
        f.write("                        'MarkerSize', markerSize, ...\n")
        f.write("                        'Color', customColors(colorIdx,:));\n")
        f.write("                    plot_handles = [plot_handles, h];\n")
        f.write("                    legend_entries{end+1} = label_text;\n")
        f.write("                end\n")
        f.write("            else\n")
        f.write("                warning(['Data length for ', varname, ' exceeds x_values length']);\n")
        f.write("            end\n")
        f.write("        end\n")
        f.write("    end\n")
        f.write("end\n\n")
        
        f.write("%% 添加标签和调整样式 %%\n")
        f.write("% 添加X轴标签\n")
        f.write("if exist('x_label', 'var')\n")
        f.write("    if ischar(x_label) || iscellstr(x_label)\n")
        f.write("        xlabel(x_label, 'FontSize', labelFontSize);\n")
        f.write("    end\n")
        f.write("end\n\n")
        
        f.write("% 添加Y轴标签\n")
        f.write("if exist('y_label', 'var')\n")
        f.write("    if ischar(y_label) || iscellstr(y_label)\n")
        f.write("        ylabel(y_label, 'FontSize', labelFontSize);\n")
        f.write("    end\n")
        f.write("end\n\n")
        
        f.write("% 添加标题\n")
        f.write("if exist('title_text', 'var')\n")
        f.write("    try\n")
        f.write("        if ischar(title_text) || iscellstr(title_text)\n")
        f.write("            title_str = title_text;\n")
        f.write("            if iscellstr(title_text) && ~isempty(title_text)\n")
        f.write("                title_str = title_text{1};\n")
        f.write("            end\n")
        f.write("            title(title_str, 'FontSize', titleFontSize);\n")
        f.write("        else\n")
        f.write("            disp('标题不是字符串或单元格数组，跳过');\n")
        f.write("        end\n")
        f.write("    catch err\n")
        f.write("        disp(['设置标题时出错: ', err.message]);\n")
        f.write("    end\n")
        f.write("end\n\n")
        
        f.write("% 创建图例\n")
        f.write("if ~isempty(plot_handles) && ~isempty(legend_entries)\n")
        f.write("    try\n")
        f.write("        lgd = legend(plot_handles, legend_entries);\n")
        f.write("        set(lgd, 'FontSize', legendFontSize, 'Location', legendLocation);\n")
        f.write("    catch err\n")
        f.write("        disp(['创建图例时出错: ', err.message]);\n")
        f.write("    end\n")
        f.write("end\n\n")
        
        f.write("% 添加网格\n")
        f.write("grid on;\n")
        f.write("set(gca, 'GridAlpha', gridAlpha, 'GridLineStyle', gridStyle);\n")
        f.write("box on;\n")
        f.write("hold off;\n\n")
        
        f.write("%% 调整图表大小和边距 %%\n")
        f.write("ax = gca;\n")
        f.write("ax.FontSize = axesFontSize;\n")
        f.write("set(gcf, 'Position', [100, 100, figWidth, figHeight]);\n")
        f.write("set(gca, 'LooseInset', get(gca, 'TightInset'));\n\n")
        
        f.write("%% 保存图表 %%\n")
        f.write("if saveFigure\n")
        f.write("    figFileName = '")
        f.write(os.path.basename(output_path).replace('.mat', ''))
        f.write("';\n")
        f.write("    try\n")
        f.write("        saveas(gcf, [figFileName, '.', outputFormat]);\n")
        f.write("        disp(['图表已保存为: ', figFileName, '.', outputFormat]);\n")
        f.write("        \n")
        f.write("        % 如果需要，可以同时保存为其他格式\n")
        f.write("        % print(gcf, figFileName, '-dpng', ['-r', num2str(figureResolution)]);\n")
        f.write("        % print(gcf, figFileName, '-depsc');\n")
        f.write("        % print(gcf, figFileName, '-dpdf');\n")
        f.write("    catch err\n")
        f.write("        disp(['保存图表时出错: ', err.message]);\n")
        f.write("    end\n")
        f.write("end\n\n")
        
        f.write("%% 完成 %%\n")
        f.write("disp('图表创建完成！');\n")
    
    print(f"MATLAB脚本已创建：{script_path}")

def plot_vs_velocity(results_list, labels, metric, fixed_snr, output_dir, save_format, se_type="both", save_matlab=False):
    """Plot metric vs velocity at a fixed SNR for multiple result sets"""
    plt.figure(figsize=(10, 6))
    
    # Color palette for multiple lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    # For MATLAB export
    all_velocities = []
    all_values = []
    all_labels = []
    
    # If we're plotting SE and true values are needed, extract true values first
    if metric == 'se' and se_type in ["true", "both"]:
        # Collect true SE values (only need first result set as ground truth should be same for all)
        if len(results_list) > 0:
            true_velocities = []
            true_values = []
            
            for (v, s), value in sorted(results_list[0]['se']['true']['avg'].items()):
                if s == fixed_snr:
                    true_velocities.append(v)
                    true_values.append(value)
            
            if true_velocities and se_type in ["true", "both"]:
                plt.plot(true_velocities, true_values, 's--', linewidth=2, markersize=8, 
                        color='k', alpha=0.8, label='Perfect CSI')
                all_velocities = true_velocities
                all_values.append(true_values)
                all_labels.append('Perfect CSI')
    
    # Now plot estimated values for each result set
    for idx, (results, label) in enumerate(zip(results_list, labels)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        if metric == 'nmse':
            # Extract NMSE data
            velocities = []
            values = []
            for (v, s), value in sorted(results['nmse']['avg'].items()):
                if s == fixed_snr:
                    velocities.append(v)
                    values.append(value)
            
            if velocities:
                values_db = nmse_to_db(np.array(values))
                plt.plot(velocities, values_db, marker=marker, linestyle='-', linewidth=2, markersize=8, 
                        color=color, label=f'{label}')
                all_velocities = velocities
                all_values.append(values_db)
                all_labels.append(label)
                
        elif metric == 'se' and se_type in ["estimated", "both"]:
            # Extract SE data
            velocities = []
            est_values = []
            
            for (v, s), value in sorted(results['se']['est']['avg'].items()):
                if s == fixed_snr:
                    velocities.append(v)
                    est_values.append(value)
            
            if velocities:
                plt.plot(velocities, est_values, marker=marker, linestyle='-', linewidth=2, markersize=8, 
                        color=color, label=f'{label}')
                if not all_velocities:
                    all_velocities = velocities
                all_values.append(est_values)
                all_labels.append(label)
    
    if metric == 'nmse':
        plt.ylabel('NMSE (dB)', fontsize=14)
        plt.title(f'NMSE vs Velocity at SNR={fixed_snr} dB', fontsize=16)
        y_label = 'NMSE (dB)'
        title = f'NMSE vs Velocity at SNR={fixed_snr} dB'
    elif metric == 'se':
        plt.ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=14)
        plt.title(f'Spectral Efficiency vs Velocity at SNR={fixed_snr} dB', fontsize=16)
        y_label = 'Spectral Efficiency (bits/s/Hz)'
        title = f'Spectral Efficiency vs Velocity at SNR={fixed_snr} dB'
    
    plt.xlabel('Velocity (km/h)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{metric}_vs_velocity_snr{int(fixed_snr)}.{save_format}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Save MATLAB data if requested
    if save_matlab and all_velocities:
        matlab_data = {
            'x_values': np.array(all_velocities),
            'x_label': 'Velocity (km/h)',
            'y_label': y_label,
            'title_text': title,
            'labels': all_labels
        }
        
        # Add each line's y values
        for i, values in enumerate(all_values):
            matlab_data[f'y_values_{i}'] = np.array(values)
        
        matlab_path = os.path.join(output_dir, f"{metric}_vs_velocity_snr{int(fixed_snr)}.mat")
        save_matlab_data(matlab_data, matlab_path)
        print(f"MATLAB data saved to {matlab_path}")
    
    plt.close()

def plot_vs_snr(results_list, labels, metric, fixed_velocity, output_dir, save_format, se_type="both", save_matlab=False):
    """Plot metric vs SNR at a fixed velocity for multiple result sets"""
    plt.figure(figsize=(10, 6))
    
    # Color palette for multiple lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    # For MATLAB export
    all_snrs = []
    all_values = []
    all_labels = []
    
    # If we're plotting SE and true values are needed, extract true values first
    if metric == 'se' and se_type in ["true", "both"]:
        # Collect true SE values (only need first result set as ground truth should be same for all)
        if len(results_list) > 0:
            true_snrs = []
            true_values = []
            
            for (v, s), value in sorted(results_list[0]['se']['true']['avg'].items()):
                if v == fixed_velocity:
                    true_snrs.append(s)
                    true_values.append(value)
            
            if true_snrs and se_type in ["true", "both"]:
                plt.plot(true_snrs, true_values, 's--', linewidth=2, markersize=8, 
                        color='k', alpha=0.8, label='Perfect CSI')
                all_snrs = true_snrs
                all_values.append(true_values)
                all_labels.append('Perfect CSI')
    
    # Now plot estimated values for each result set
    for idx, (results, label) in enumerate(zip(results_list, labels)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        if metric == 'nmse':
            # Extract NMSE data
            snrs = []
            values = []
            for (v, s), value in sorted(results['nmse']['avg'].items()):
                if v == fixed_velocity:
                    snrs.append(s)
                    values.append(value)
            
            if snrs:
                values_db = nmse_to_db(np.array(values))
                plt.plot(snrs, values_db, marker=marker, linestyle='-', linewidth=2, markersize=8, 
                        color=color, label=f'{label}')
                all_snrs = snrs
                all_values.append(values_db)
                all_labels.append(label)
                
        elif metric == 'se' and se_type in ["estimated", "both"]:
            # Extract SE data
            snrs = []
            est_values = []
            
            for (v, s), value in sorted(results['se']['est']['avg'].items()):
                if v == fixed_velocity:
                    snrs.append(s)
                    est_values.append(value)
            
            if snrs:
                plt.plot(snrs, est_values, marker=marker, linestyle='-', linewidth=2, markersize=8, 
                        color=color, label=f'{label}')
                if not all_snrs:
                    all_snrs = snrs
                all_values.append(est_values)
                all_labels.append(label)
    
    if metric == 'nmse':
        plt.ylabel('NMSE (dB)', fontsize=14)
        plt.title(f'NMSE vs SNR at Velocity={fixed_velocity} km/h', fontsize=16)
        y_label = 'NMSE (dB)'
        title = f'NMSE vs SNR at Velocity={fixed_velocity} km/h'
    elif metric == 'se':
        plt.ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=14)
        plt.title(f'Spectral Efficiency vs SNR at Velocity={fixed_velocity} km/h', fontsize=16)
        y_label = 'Spectral Efficiency (bits/s/Hz)'
        title = f'Spectral Efficiency vs SNR at Velocity={fixed_velocity} km/h'
    
    plt.xlabel('SNR (dB)', fontsize=14)
    
    # 设置x轴刻度，只显示整数SNR值
    if 'snrs' in locals() and snrs:
        min_snr = int(min(snrs))
        max_snr = int(max(snrs)) + 1
        plt.xticks(range(min_snr, max_snr))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{metric}_vs_snr_vel{fixed_velocity}.{save_format}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Save MATLAB data if requested
    if save_matlab and all_snrs:
        matlab_data = {
            'x_values': np.array(all_snrs),
            'x_label': 'SNR (dB)',
            'y_label': y_label,
            'title_text': title,
            'labels': all_labels
        }
        
        # Add each line's y values
        for i, values in enumerate(all_values):
            matlab_data[f'y_values_{i}'] = np.array(values)
        
        matlab_path = os.path.join(output_dir, f"{metric}_vs_snr_vel{fixed_velocity}.mat")
        save_matlab_data(matlab_data, matlab_path)
        print(f"MATLAB data saved to {matlab_path}")
    
    plt.close()

def plot_vs_step(results_list, labels, metric, fixed_velocity, fixed_snr, output_dir, save_format, se_type="both", frame_interval=0.625, save_matlab=False):
    """Plot metric vs data frame for a specific (velocity, SNR) pair for multiple result sets"""
    plt.figure(figsize=(10, 6))
    
    # Color palette for multiple lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    key = (fixed_velocity, fixed_snr)
    has_data = False
    
    # For MATLAB export
    all_frames = []
    all_values = []
    all_labels = []
    
    # If we're plotting SE and true values are needed, extract true values first
    if metric == 'se' and se_type in ["true", "both"]:
        # Collect true SE values (only need first result set as ground truth should be same for all)
        if len(results_list) > 0 and key in results_list[0]['se']['true']['per_step']:
            true_values = results_list[0]['se']['true']['per_step'][key]
            frames = np.arange(1, len(true_values) + 1)
            
            plt.plot(frames, true_values, 's--', linewidth=2, markersize=8, 
                     color='k', alpha=0.8, label='Perfect CSI')
            has_data = True
            all_frames = frames
            all_values.append(true_values)
            all_labels.append('Perfect CSI')
    
    # Now plot for each result set
    for idx, (results, label) in enumerate(zip(results_list, labels)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        if metric == 'nmse':
            if key in results['nmse']['per_step']:
                values = results['nmse']['per_step'][key]
                frames = np.arange(1, len(values) + 1)
                
                values_db = nmse_to_db(np.array(values))
                plt.plot(frames, values_db, marker=marker, linestyle='-', linewidth=2, markersize=8, 
                        color=color, label=f'{label}')
                has_data = True
                all_frames = frames
                all_values.append(values_db)
                all_labels.append(label)
        
        elif metric == 'se' and se_type in ["estimated", "both"]:
            if key in results['se']['est']['per_step']:
                est_values = results['se']['est']['per_step'][key]
                frames = np.arange(1, len(est_values) + 1)
                
                plt.plot(frames, est_values, marker=marker, linestyle='-', linewidth=2, markersize=8, 
                        color=color, label=f'{label}')
                has_data = True
                if not all_frames.size:
                    all_frames = frames
                all_values.append(est_values)
                all_labels.append(label)
    
    if not has_data:
        print(f"No data available for {metric} vs frame at velocity={fixed_velocity}, SNR={fixed_snr}")
        plt.close()
        return
        
    if metric == 'nmse':
        plt.ylabel('NMSE (dB)', fontsize=14)
        plt.title(f'NMSE vs Data Frame at V={fixed_velocity} km/h, SNR={fixed_snr} dB', fontsize=16)
        y_label = 'NMSE (dB)'
        title = f'NMSE vs Data Frame at V={fixed_velocity} km/h, SNR={fixed_snr} dB'
    elif metric == 'se':
        plt.ylabel('Spectral Efficiency (bits/s/Hz)', fontsize=14)
        plt.title(f'SE vs Data Frame at V={fixed_velocity} km/h, SNR={fixed_snr} dB', fontsize=16)
        y_label = 'Spectral Efficiency (bits/s/Hz)'
        title = f'SE vs Data Frame at V={fixed_velocity} km/h, SNR={fixed_snr} dB'
    
    plt.xlabel(f'Data Frame ({frame_interval}ms)', fontsize=14)
    x_label = f'Data Frame ({frame_interval}ms)'
    
    # Show integer frame numbers on x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{metric}_vs_frame_vel{fixed_velocity}_snr{int(fixed_snr)}.{save_format}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Save MATLAB data if requested
    if save_matlab and all_frames.size:
        matlab_data = {
            'x_values': all_frames,
            'x_label': x_label,
            'y_label': y_label,
            'title_text': title,
            'labels': all_labels
        }
        
        # Add each line's y values
        for i, values in enumerate(all_values):
            matlab_data[f'y_values_{i}'] = np.array(values)
        
        matlab_path = os.path.join(output_dir, f"{metric}_vs_frame_vel{fixed_velocity}_snr{int(fixed_snr)}.mat")
        save_matlab_data(matlab_data, matlab_path)
        print(f"MATLAB data saved to {matlab_path}")
    
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all result files
    results_list = []
    
    for result_path in args.results:
        print(f"Loading results from {result_path}")
        results = load_results(result_path)
        results_list.append(results)
    
    # Create labels if not provided
    if args.labels is None or len(args.labels) != len(args.results):
        labels = [f"Model {i+1}" for i in range(len(args.results))]
    else:
        labels = args.labels
    
    # Determine which plots to generate
    if args.plot_type == "all" or args.plot_type == "velocity":
        if args.metric == "nmse" or args.metric == "both":
            print(f"Generating NMSE vs velocity plot (SNR = {args.snr} dB)")
            plot_vs_velocity(results_list, labels, "nmse", args.snr, args.output_dir, args.save_format, save_matlab=args.save_matlab)
            
        if args.metric == "se" or args.metric == "both":
            print(f"Generating SE vs velocity plot (SNR = {args.snr} dB)")
            plot_vs_velocity(results_list, labels, "se", args.snr, args.output_dir, args.save_format, args.se_type, save_matlab=args.save_matlab)
    
    if args.plot_type == "all" or args.plot_type == "snr":
        if args.metric == "nmse" or args.metric == "both":
            print(f"Generating NMSE vs SNR plot (velocity = {args.velocity} km/h)")
            plot_vs_snr(results_list, labels, "nmse", args.velocity, args.output_dir, args.save_format, save_matlab=args.save_matlab)
            
        if args.metric == "se" or args.metric == "both":
            print(f"Generating SE vs SNR plot (velocity = {args.velocity} km/h)")
            plot_vs_snr(results_list, labels, "se", args.velocity, args.output_dir, args.save_format, args.se_type, save_matlab=args.save_matlab)
    
    if args.plot_type == "all" or args.plot_type == "step":
        if args.metric == "nmse" or args.metric == "both":
            print(f"Generating NMSE vs data frame plot (velocity = {args.velocity} km/h, SNR = {args.snr} dB)")
            plot_vs_step(results_list, labels, "nmse", args.velocity, args.snr, args.output_dir, args.save_format, 
                        frame_interval=args.frame_interval, save_matlab=args.save_matlab)
            
        if args.metric == "se" or args.metric == "both":
            print(f"Generating SE vs data frame plot (velocity = {args.velocity} km/h, SNR = {args.snr} dB)")
            plot_vs_step(results_list, labels, "se", args.velocity, args.snr, args.output_dir, args.save_format, 
                        args.se_type, frame_interval=args.frame_interval, save_matlab=args.save_matlab)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 
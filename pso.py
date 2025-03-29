# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
#
# # 参数设置
# MAP_SIZE = 500  # 地图大小 100x100
# NUM_USERS = 2000  # 用户数量
# MAX_RADIUS = 15  # 基站最大覆盖半径
# MIN_RADIUS = 5  # 基站最小覆盖半径
# MAX_STATIONS = 20  # 最大基站数量
#
# # 随机生成用户位置
# def generate_users_from_image(image_path, num_users=2000):
#     """根据图片生成用户分布"""
#     img = Image.open(image_path).convert('L')  # 转为灰度图
#     img = img.resize((MAP_SIZE, MAP_SIZE))  # 调整为地图大小
#
#     # 将灰度图转为数组
#     density = np.array(img)
#
#     # 将灰度值归一化为权重
#     weights = density / np.sum(density)
#
#     # 随机生成用户位置，按权重采样
#     x, y = np.meshgrid(np.arange(MAP_SIZE), np.arange(MAP_SIZE))
#     coords = np.vstack((x.ravel(), y.ravel())).T
#
#     # 根据灰度值权重随机采样用户位置
#     selected_indices = np.random.choice(len(coords), size=num_users, p=weights.ravel())
#     users = coords[selected_indices]
#
#     return users
#
#
# image_path = "renkoumidu.jpg"  # 替换为你的地图图像路径
# users = generate_users_from_image(image_path, NUM_USERS)
#
#
# def plot_map_with_density(stations, users, title, coverage, num_stations, image_path):
#     """绘制带有人口密度的基站覆盖图"""
#     img = Image.open(image_path).convert('L')
#     img = img.resize((MAP_SIZE, MAP_SIZE))
#
#     plt.figure(figsize=(10, 10))
#
#     # 显示人口密度图
#     plt.imshow(img, cmap='gray', extent=[0, MAP_SIZE, 0, MAP_SIZE], origin='upper')
#
#     # 显示用户位置
#     plt.scatter(users[:, 0], users[:, 1], c='blue', label='Users', s=5)
#
#     # 显示基站覆盖
#     for (x, y, r) in stations:
#         circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
#         plt.gca().add_patch(circle)
#
#     plt.title(f"{title}\nCoverage: {coverage:.2%}, Base Stations: {num_stations}")
#     plt.xlabel("X-coordinate")
#     plt.ylabel("Y-coordinate")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#
#
#
# # 计算覆盖率
# def coverage_rate(stations, users):
#     covered = np.zeros(len(users), dtype=bool)
#     for (x, y, r) in stations:
#         dist = np.linalg.norm(users - np.array([x, y]), axis=1)
#         covered |= (dist <= r)
#     return np.sum(covered) / len(users)
#
#
# # 绘制地图
# def plot_map(stations, users, title, coverage, num_stations):
#     plt.figure(figsize=(10, 10))
#     plt.scatter(users[:, 0], users[:, 1], c='blue', label='Users', s=5)
#
#     for (x, y, r) in stations:
#         circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
#         plt.gca().add_patch(circle)
#
#     plt.title(f"{title}\nCoverage: {coverage:.2%}, Base Stations: {num_stations}")
#     plt.xlabel("X-coordinate")
#     plt.ylabel("Y-coordinate")
#     plt.grid(True)
#     plt.legend()
#     plt.show()
#
#
# # ========== 模拟退火算法 ========== #
# def simulated_annealing(users, max_iter=100, initial_temp=1000, cooling_rate=0.995):
#     stations = np.random.uniform(0, MAP_SIZE, (MAX_STATIONS, 3))
#     stations[:, 2] = np.random.uniform(MIN_RADIUS, MAX_RADIUS, len(stations))
#
#     best_stations = stations.copy()
#     best_coverage = coverage_rate(best_stations, users)
#
#     temp = initial_temp
#     for step in range(max_iter):
#         new_stations = best_stations.copy()
#         idx = np.random.randint(len(new_stations))
#
#         # 随机扰动基站位置和半径
#         new_stations[idx, :2] += np.random.uniform(-5, 5, 2)
#         new_stations[idx, 2] = np.clip(new_stations[idx, 2] + np.random.uniform(-2, 2), MIN_RADIUS, MAX_RADIUS)
#         new_stations = np.clip(new_stations, 0, MAP_SIZE)
#
#         # 计算覆盖率
#         new_coverage = coverage_rate(new_stations, users)
#
#         # 模拟退火概率
#         delta = new_coverage - best_coverage
#         if delta > 0 or np.exp(delta / temp) > np.random.rand():
#             best_stations = new_stations
#             best_coverage = new_coverage
#
#         # 降温
#         temp *= cooling_rate
#
#     return best_stations, best_coverage
#
#
# # ========== 粒子群算法 ========== #
# def particle_swarm(users, num_particles=30, max_iter=100):
#     particles = np.random.uniform(0, MAP_SIZE, (num_particles, MAX_STATIONS, 3))
#     particles[:, :, 2] = np.random.uniform(MIN_RADIUS, MAX_RADIUS, (num_particles, MAX_STATIONS))
#
#     velocities = np.random.uniform(-1, 1, (num_particles, MAX_STATIONS, 3))
#     pbest = particles.copy()
#     gbest = particles[np.argmax([coverage_rate(p, users) for p in particles])]
#
#     inertia = 0.7  # 惯性权重
#     c1, c2 = 1.5, 1.5  # 学习因子
#
#     for _ in range(max_iter):
#         for i in range(num_particles):
#             velocities[i] = (inertia * velocities[i] +
#                              c1 * np.random.rand() * (pbest[i] - particles[i]) +
#                              c2 * np.random.rand() * (gbest - particles[i]))
#
#             particles[i] += velocities[i]
#
#             # 边界处理
#             particles[i][:, :2] = np.clip(particles[i][:, :2], 0, MAP_SIZE)
#             particles[i][:, 2] = np.clip(particles[i][:, 2], MIN_RADIUS, MAX_RADIUS)
#
#             # 更新pbest
#             if coverage_rate(particles[i], users) > coverage_rate(pbest[i], users):
#                 pbest[i] = particles[i]
#
#         # 更新gbest
#         gbest = pbest[np.argmax([coverage_rate(p, users) for p in pbest])]
#
#     return gbest, coverage_rate(gbest, users)
#
#
# # ========== 多次运行取平均覆盖率 ========== #
# def run_multiple_trials(users, trials=10):
#     sa_results, pso_results = [], []
#
#     for _ in range(trials):
#         sa_stations, sa_coverage = simulated_annealing(users)
#         sa_results.append(sa_coverage)
#
#         pso_stations, pso_coverage = particle_swarm(users)
#         pso_results.append(pso_coverage)
#
#     print(f"模拟退火平均覆盖率: {np.mean(sa_results):.2%}")
#     print(f"粒子群平均覆盖率: {np.mean(pso_results):.2%}")
#
#     plot_map(sa_stations, users, "Simulated Annealing", np.mean(sa_results), len(sa_stations))
#     plot_map(pso_stations, users, "Particle Swarm Optimization", np.mean(pso_results), len(pso_stations))
#
#
# # ========== 主函数运行 ========== #
# # ========== 主函数运行 ========== #
# if __name__ == "__main__":
#     image_path = "renkoumidu.jpg"  # 替换为你的地图路径
#     users = generate_users_from_image(image_path, NUM_USERS)
#
#     run_multiple_trials(users)  # 多次运行算法
#
#     sa_stations, sa_coverage = simulated_annealing(users)
#     pso_stations, pso_coverage = particle_swarm(users)
#
#     # 绘制带人口密度的地图
#     plot_map_with_density(sa_stations, users, "Simulated Annealing", sa_coverage, len(sa_stations), image_path)
#     plot_map_with_density(pso_stations, users, "Particle Swarm Optimization", pso_coverage, len(pso_stations),
#                           image_path)


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import logging
# import time
#
# # ========== 日志配置 ==========
# LOG_FILE = "base_station_optimization.log"
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# console.setFormatter(formatter)
# logging.getLogger("").addHandler(console)
#
# # ========== 参数设置 ==========
# NUM_USERS = 2000
# NUM_BASE_STATIONS = 30
# WHITE_THRESHOLD = 250
# MAX_ITERATIONS = 100
# SA_TEMPERATURE = 1000
# SA_COOLING_RATE = 0.95
# PSO_PARTICLES = 50
# PSO_ITERATIONS = 100
#
#
# # ========== 数据生成 ==========
# def generate_users_exclude_white(image_path, num_users=2000, white_threshold=WHITE_THRESHOLD):
#     """生成用户分布，仅在非白色区域"""
#     logging.info("加载图片: %s", image_path)
#
#     img = Image.open(image_path).convert('L')
#     width, height = img.size
#     density = np.array(img)
#
#     # 标记非白色区域
#     non_white_mask = density < white_threshold
#     weights = np.zeros_like(density, dtype=float)
#     weights[non_white_mask] = 255 - density[non_white_mask]
#
#     if np.sum(weights) == 0:
#         raise ValueError("图片几乎全白，无有效区域")
#
#     weights /= np.sum(weights)
#
#     # 生成用户位置
#     y_coords, x_coords = np.mgrid[:height, :width]
#     coords = np.vstack((x_coords.ravel(), y_coords.ravel())).T
#     selected_indices = np.random.choice(len(coords), size=num_users, p=weights.ravel())
#     users = coords[selected_indices]
#
#     logging.info("生成用户数量: %d", num_users)
#     return users, width, height, weights
#
#
# # ========== 生成基站 ==========
# def generate_valid_base_stations(weights, num_base_stations):
#     """在非白色区域生成基站"""
#     coords = np.column_stack(np.where(weights > 0))
#     selected_indices = np.random.choice(len(coords), size=num_base_stations, replace=False)
#     base_stations = coords[selected_indices]
#
#     logging.info("生成基站数量: %d", num_base_stations)
#     return base_stations
#
#
# # ========== 目标函数 ==========
# def coverage_score(base_stations, users, radius=30):
#     """计算基站对用户的覆盖率"""
#     covered_users = 0
#     for x, y in users:
#         distances = np.linalg.norm(base_stations - np.array([x, y]), axis=1)
#         if np.any(distances <= radius):
#             covered_users += 1
#     coverage = covered_users / len(users)
#
#     logging.info("覆盖率: %.4f", coverage)
#     return coverage
#
#
# # ========== 模拟退火算法 SA ==========
# def sa_optimize(users, width, height, weights, num_base_stations=NUM_BASE_STATIONS, radius=30):
#     """模拟退火算法进行基站优化"""
#
#     logging.info("SA优化开始")
#     start_time = time.time()
#
#     base_stations = generate_valid_base_stations(weights, num_base_stations)
#
#     best_score = coverage_score(base_stations, users, radius)
#     best_solution = base_stations.copy()
#
#     temperature = SA_TEMPERATURE
#
#     for i in range(MAX_ITERATIONS):
#         new_base_stations = generate_valid_base_stations(weights, num_base_stations)
#         new_score = coverage_score(new_base_stations, users, radius)
#
#         # 接受新解
#         if new_score > best_score or np.random.rand() < np.exp((new_score - best_score) / temperature):
#             best_score = new_score
#             best_solution = new_base_stations.copy()
#
#         temperature *= SA_COOLING_RATE
#
#         logging.info("SA迭代 %d/%d -> 覆盖率: %.4f", i + 1, MAX_ITERATIONS, best_score)
#
#     end_time = time.time()
#     logging.info("SA优化完成, 耗时: %.2f 秒", end_time - start_time)
#
#     return best_solution
#
#
# # ========== 粒子群算法 PSO ==========
# def generate_valid_particles(weights, num_particles, num_base_stations):
#     """在非白色区域生成粒子群"""
#     coords = np.column_stack(np.where(weights > 0))
#     particles = np.empty((num_particles, num_base_stations, 2), dtype=int)
#
#     for i in range(num_particles):
#         selected_indices = np.random.choice(len(coords), size=num_base_stations, replace=False)
#         particles[i] = coords[selected_indices]
#
#     logging.info("生成粒子群: %d 个粒子", num_particles)
#     return particles
#
#
# def pso_optimize(users, width, height, weights, num_base_stations=NUM_BASE_STATIONS, radius=30):
#     """粒子群算法进行基站优化"""
#
#     logging.info("PSO优化开始")
#     start_time = time.time()
#
#     particles = generate_valid_particles(weights, PSO_PARTICLES, num_base_stations)
#
#     best_positions = particles.copy()
#     best_scores = np.array([coverage_score(p, users, radius) for p in particles])
#
#     global_best_idx = np.argmax(best_scores)
#     global_best = particles[global_best_idx]
#
#     for i in range(PSO_ITERATIONS):
#         new_particles = generate_valid_particles(weights, PSO_PARTICLES, num_base_stations)
#
#         scores = np.array([coverage_score(p, users, radius) for p in new_particles])
#
#         better_mask = scores > best_scores
#         best_scores[better_mask] = scores[better_mask]
#         best_positions[better_mask] = new_particles[better_mask]
#
#         global_best_idx = np.argmax(best_scores)
#         global_best = best_positions[global_best_idx]
#
#         logging.info("PSO迭代 %d/%d -> 覆盖率: %.4f", i + 1, PSO_ITERATIONS, best_scores[global_best_idx])
#
#     end_time = time.time()
#     logging.info("PSO优化完成, 耗时: %.2f 秒", end_time - start_time)
#
#     return global_best
#
#
# # ========== 可视化 ==========
# def plot_results(users, base_stations_sa, base_stations_pso, image_path, width, height):
#     """绘制基站分布图"""
#     logging.info("开始可视化")
#
#     img = Image.open(image_path).convert('L')
#     plt.figure(figsize=(14, 7))
#
#     # SA结果
#     plt.subplot(1, 2, 1)
#     plt.imshow(img, cmap='gray', extent=[0, width, 0, height], origin='upper')
#     plt.scatter(users[:, 0], height - users[:, 1], c='blue', s=5, label='Users')
#     plt.scatter(base_stations_sa[:, 0], height - base_stations_sa[:, 1], c='red', label='SA Base Stations', s=30)
#     plt.title("SA Base Station Distribution")
#     plt.legend()
#
#     # PSO结果
#     plt.subplot(1, 2, 2)
#     plt.imshow(img, cmap='gray', extent=[0, width, 0, height], origin='upper')
#     plt.scatter(users[:, 0], height - users[:, 1], c='blue', s=5, label='Users')
#     plt.scatter(base_stations_pso[:, 0], height - base_stations_pso[:, 1], c='green', label='PSO Base Stations', s=30)
#     plt.title("PSO Base Station Distribution")
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#     logging.info("可视化完成")
#
#
# # ========== 主程序运行 ==========
# if __name__ == "__main__":
#     image_path = "renkoumidu.jpg"
#
#     users, width, height, weights = generate_users_exclude_white(image_path, NUM_USERS)
#
#     base_stations_sa = sa_optimize(users, width, height, weights)
#     base_stations_pso = pso_optimize(users, width, height, weights)
#
#     plot_results(users, base_stations_sa, base_stations_pso, image_path, width, height)
#
#     logging.info("✅ 程序运行完成")

# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import logging
# import time
#
# # ========== 日志配置 ==========
# LOG_FILE = "base_station_optimization.log"
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
# console.setFormatter(formatter)
# logging.getLogger("").addHandler(console)
#
# # ========== 参数设置 ==========
# NUM_USERS = 2000
# NUM_BASE_STATIONS = 30
# WHITE_THRESHOLD = 250
# MAX_ITERATIONS = 100
# SA_TEMPERATURE = 1000
# SA_COOLING_RATE = 0.95
# PSO_PARTICLES = 50
# PSO_ITERATIONS = 100
#
#
# # ========== 数据生成 ==========
# def generate_users_exclude_white(image_path, num_users=2000, white_threshold=WHITE_THRESHOLD):
#     """生成用户分布，仅在非白色区域"""
#     logging.info("加载图片: %s", image_path)
#
#     img = Image.open(image_path).convert('L')
#     width, height = img.size
#     density = np.array(img)
#
#     # 标记非白色区域
#     non_white_mask = density < white_threshold
#     weights = np.zeros_like(density, dtype=float)
#     weights[non_white_mask] = 255 - density[non_white_mask]
#
#     if np.sum(weights) == 0:
#         raise ValueError("图片几乎全白，无有效区域")
#
#     weights /= np.sum(weights)
#
#     # 生成用户位置
#     y_coords, x_coords = np.mgrid[:height, :width]
#     coords = np.vstack((x_coords.ravel(), y_coords.ravel())).T
#     selected_indices = np.random.choice(len(coords), size=num_users, p=weights.ravel())
#     users = coords[selected_indices]
#
#     logging.info("生成用户数量: %d", num_users)
#     return users, width, height, weights
#
#
# # ========== 强制生成非白色基站 ==========
# def generate_valid_base_stations(weights, num_base_stations):
#     """在非白色区域生成基站（强制排除白色区域）"""
#     height, width = weights.shape
#     coords = np.column_stack(np.where(weights > 0))  # 非白色区域
#
#     base_stations = []
#     retry_count = 0
#
#     while len(base_stations) < num_base_stations:
#         idx = np.random.choice(len(coords))
#         x, y = coords[idx]
#
#         # 检查是否在白色区域
#         if weights[y, x] > 0:
#             base_stations.append((x, y))
#         else:
#             retry_count += 1
#
#     logging.info("生成基站数量: %d (重试: %d 次)", num_base_stations, retry_count)
#     return np.array(base_stations)
#
#
# # ========== 强制生成非白色粒子 ==========
# def generate_valid_particles(weights, num_particles, num_base_stations):
#     """在非白色区域生成粒子群（强制排除白色区域）"""
#     coords = np.column_stack(np.where(weights > 0))
#     particles = np.empty((num_particles, num_base_stations, 2), dtype=int)
#
#     for i in range(num_particles):
#         retry_count = 0
#         particle = []
#
#         while len(particle) < num_base_stations:
#             idx = np.random.choice(len(coords))
#             x, y = coords[idx]
#
#             if weights[y, x] > 0:  # 确保在非白色区域
#                 particle.append((x, y))
#             else:
#                 retry_count += 1
#
#         particles[i] = np.array(particle)
#
#         logging.info("粒子 %d 生成完成 (重试: %d 次)", i + 1, retry_count)
#
#     return particles
#
#
# # ========== 目标函数 ==========
# def coverage_score(base_stations, users, radius=30):
#     """计算基站对用户的覆盖率"""
#     covered_users = 0
#     for x, y in users:
#         distances = np.linalg.norm(base_stations - np.array([x, y]), axis=1)
#         if np.any(distances <= radius):
#             covered_users += 1
#     coverage = covered_users / len(users)
#
#     logging.info("覆盖率: %.4f", coverage)
#     return coverage
#
#
# # ========== 模拟退火算法 SA ==========
# def sa_optimize(users, width, height, weights, num_base_stations=NUM_BASE_STATIONS, radius=30):
#     """模拟退火算法进行基站优化"""
#
#     logging.info("SA优化开始")
#     start_time = time.time()
#
#     base_stations = generate_valid_base_stations(weights, num_base_stations)
#
#     best_score = coverage_score(base_stations, users, radius)
#     best_solution = base_stations.copy()
#
#     temperature = SA_TEMPERATURE
#
#     for i in range(MAX_ITERATIONS):
#         new_base_stations = generate_valid_base_stations(weights, num_base_stations)
#         new_score = coverage_score(new_base_stations, users, radius)
#
#         if new_score > best_score or np.random.rand() < np.exp((new_score - best_score) / temperature):
#             best_score = new_score
#             best_solution = new_base_stations.copy()
#
#         temperature *= SA_COOLING_RATE
#
#         logging.info("SA迭代 %d/%d -> 覆盖率: %.4f", i + 1, MAX_ITERATIONS, best_score)
#
#     end_time = time.time()
#     logging.info("SA优化完成, 耗时: %.2f 秒", end_time - start_time)
#
#     return best_solution
#
#
# # ========== 粒子群算法 PSO ==========
# def pso_optimize(users, width, height, weights, num_base_stations=NUM_BASE_STATIONS, radius=30):
#     """粒子群算法进行基站优化"""
#
#     logging.info("PSO优化开始")
#     start_time = time.time()
#
#     particles = generate_valid_particles(weights, PSO_PARTICLES, num_base_stations)
#
#     best_positions = particles.copy()
#     best_scores = np.array([coverage_score(p, users, radius) for p in particles])
#
#     global_best_idx = np.argmax(best_scores)
#     global_best = particles[global_best_idx]
#
#     for i in range(PSO_ITERATIONS):
#         new_particles = generate_valid_particles(weights, PSO_PARTICLES, num_base_stations)
#
#         scores = np.array([coverage_score(p, users, radius) for p in new_particles])
#
#         better_mask = scores > best_scores
#         best_scores[better_mask] = scores[better_mask]
#         best_positions[better_mask] = new_particles[better_mask]
#
#         global_best_idx = np.argmax(best_scores)
#         global_best = best_positions[global_best_idx]
#
#         logging.info("PSO迭代 %d/%d -> 覆盖率: %.4f", i + 1, PSO_ITERATIONS, best_scores[global_best_idx])
#
#     end_time = time.time()
#     logging.info("PSO优化完成, 耗时: %.2f 秒", end_time - start_time)
#
#     return global_best
#
#
# # ========== 可视化 ==========
# def plot_results(users, base_stations_sa, base_stations_pso, image_path, width, height):
#     """绘制基站分布图"""
#     img = Image.open(image_path).convert('L')
#
#     plt.figure(figsize=(14, 7))
#     plt.imshow(img, cmap='gray', extent=[0, width, 0, height], origin='upper')
#
#     plt.scatter(users[:, 0], height - users[:, 1], c='blue', s=5, label='Users')
#     plt.scatter(base_stations_sa[:, 0], height - base_stations_sa[:, 1], c='red', label='SA Base Stations', s=30)
#     plt.scatter(base_stations_pso[:, 0], height - base_stations_pso[:, 1], c='green', label='PSO Base Stations', s=30)
#
#     plt.title("Base Station Distribution")
#     plt.legend()
#     plt.show()
#
#
# # ========== 主程序运行 ==========
# if __name__ == "__main__":
#     image_path = "renkoumidu.jpg"
#
#     users, width, height, weights = generate_users_exclude_white(image_path, NUM_USERS)
#     base_stations_sa = sa_optimize(users, width, height, weights)
#     base_stations_pso = pso_optimize(users, width, height, weights)
#
#     plot_results(users, base_stations_sa, base_stations_pso, image_path, width, height)


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import time
from scipy.interpolate import interp1d

# ========== 日志配置 ==========
LOG_FILE = "base_station_optimization.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# ========== 参数设置 ==========
NUM_USERS = 2000
NUM_BASE_STATIONS = 100
WHITE_THRESHOLD = 250
MAX_ITERATIONS = 100
SA_TEMPERATURE = 1000
SA_COOLING_RATE = 0.95
PSO_PARTICLES = 50
PSO_ITERATIONS = 100


# ========== 数据生成 ==========
def generate_users_exclude_white(image_path, num_users=2000, white_threshold=WHITE_THRESHOLD):
    """生成用户分布，仅在非白色区域"""
    logging.info("加载图片: %s", image_path)

    img = Image.open(image_path).convert('L')
    width, height = img.size
    density = np.array(img)

    # 标记非白色区域
    non_white_mask = density < white_threshold
    weights = np.zeros_like(density, dtype=float)
    weights[non_white_mask] = 255 - density[non_white_mask]

    if np.sum(weights) == 0:
        raise ValueError("图片几乎全白，无有效区域")

    weights /= np.sum(weights)

    # 生成用户位置
    y_coords, x_coords = np.mgrid[:height, :width]
    coords = np.vstack((x_coords.ravel(), y_coords.ravel())).T
    selected_indices = np.random.choice(len(coords), size=num_users, p=weights.ravel())
    users = coords[selected_indices]

    logging.info("生成用户数量: %d", num_users)
    return users, width, height, weights


# ========== 强制生成非白色基站 ==========
def generate_valid_base_stations(weights, num_base_stations):
    """在非白色区域生成基站（严格限制在图片范围内）"""
    height, width = weights.shape
    coords = np.column_stack(np.where(weights > 0))  # 非白色区域

    base_stations = []
    retry_count = 0

    while len(base_stations) < num_base_stations:
        idx = np.random.choice(len(coords))
        x, y = coords[idx]

        # ✅ 边界检查，防止越界
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        # 检查是否在白色区域
        if weights[y, x] > 0:
            base_stations.append((x, y))
        else:
            retry_count += 1

    logging.info("生成基站数量: %d (重试: %d 次)", num_base_stations, retry_count)
    return np.array(base_stations)


# ========== 强制生成非白色粒子 ==========
def generate_valid_particles(weights, num_particles, num_base_stations):
    """在非白色区域生成粒子群（强制排除白色区域，并防止越界）"""
    height, width = weights.shape
    coords = np.column_stack(np.where(weights > 0))

    particles = np.empty((num_particles, num_base_stations, 2), dtype=int)

    for i in range(num_particles):
        retry_count = 0
        particle = []

        while len(particle) < num_base_stations:
            idx = np.random.choice(len(coords))
            x, y = coords[idx]

            # ✅ 防止越界
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)

            if weights[y, x] > 0:
                particle.append((x, y))
            else:
                retry_count += 1

        particles[i] = np.array(particle)
        logging.info("粒子 %d 生成完成 (重试: %d 次)", i + 1, retry_count)

    return particles


# ========== 目标函数 ==========
def coverage_score(base_stations, users, radius=30):
    """计算基站对用户的覆盖率"""
    covered_users = 0
    for x, y in users:
        distances = np.linalg.norm(base_stations - np.array([x, y]), axis=1)
        if np.any(distances <= radius):
            covered_users += 1
    coverage = covered_users / len(users)

    logging.info("覆盖率: %.4f", coverage)
    return coverage


# ========== 模拟退火算法 SA ==========
# ========== 模拟退火算法 SA ==========
def sa_optimize(users, width, height, weights, num_base_stations=NUM_BASE_STATIONS, radius=30):
    """模拟退火算法进行基站优化"""
    logging.info("SA优化开始")
    start_time = time.time()

    base_stations = generate_valid_base_stations(weights, num_base_stations)

    best_score = coverage_score(base_stations, users, radius)
    best_solution = base_stations.copy()

    temperature = SA_TEMPERATURE

    # ✅ 记录覆盖率变化
    sa_coverage_history = [best_score]

    for i in range(MAX_ITERATIONS):
        new_base_stations = generate_valid_base_stations(weights, num_base_stations)
        new_score = coverage_score(new_base_stations, users, radius)

        if new_score > best_score or np.random.rand() < np.exp((new_score - best_score) / temperature):
            best_score = new_score
            best_solution = new_base_stations.copy()

        temperature *= SA_COOLING_RATE

        # 记录当前覆盖率
        sa_coverage_history.append(best_score)

        logging.info("SA迭代 %d/%d -> 覆盖率: %.4f", i + 1, MAX_ITERATIONS, best_score)

    end_time = time.time()
    logging.info("SA优化完成, 耗时: %.2f 秒", end_time - start_time)

    return best_solution, sa_coverage_history


# ========== 粒子群算法 PSO ==========
def pso_optimize(users, width, height, weights, num_base_stations=NUM_BASE_STATIONS, radius=30):
    """粒子群算法进行基站优化"""
    logging.info("PSO优化开始")
    start_time = time.time()

    particles = generate_valid_particles(weights, PSO_PARTICLES, num_base_stations)

    best_positions = particles.copy()
    best_scores = np.array([coverage_score(p, users, radius) for p in particles])

    global_best_idx = np.argmax(best_scores)
    global_best = particles[global_best_idx]

    # ✅ 记录覆盖率变化
    pso_coverage_history = [best_scores[global_best_idx]]

    for i in range(PSO_ITERATIONS):
        new_particles = generate_valid_particles(weights, PSO_PARTICLES, num_base_stations)
        scores = np.array([coverage_score(p, users, radius) for p in new_particles])

        better_mask = scores > best_scores
        best_scores[better_mask] = scores[better_mask]
        best_positions[better_mask] = new_particles[better_mask]

        global_best_idx = np.argmax(best_scores)
        global_best = best_positions[global_best_idx]

        # 记录当前覆盖率
        pso_coverage_history.append(best_scores[global_best_idx])

        logging.info("PSO迭代 %d/%d -> 覆盖率: %.4f", i + 1, PSO_ITERATIONS, best_scores[global_best_idx])

    end_time = time.time()
    logging.info("PSO优化完成, 耗时: %.2f 秒", end_time - start_time)

    return global_best, pso_coverage_history


# ========== 可视化 ==========
def plot_results(users, base_stations_sa, base_stations_pso, image_path, width, height):
    """绘制基站分布图"""
    img = Image.open(image_path).convert('L')

    plt.figure(figsize=(14, 7))

    # ========== SA结果 ==========
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', extent=[0, width, 0, height], origin='upper')  # 上方为原点，修复对齐
    plt.scatter(users[:, 0], height - users[:, 1], c='blue', s=5, label='Users')   # 修复用户位置
    plt.scatter(base_stations_sa[:, 0], height - base_stations_sa[:, 1], c='red', label='SA Base Stations', s=30)
    plt.legend()
    plt.title("SA Base Station Distribution")

    # ========== PSO结果 ==========
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray', extent=[0, width, 0, height], origin='upper')  # 上方为原点
    plt.scatter(users[:, 0], height - users[:, 1], c='blue', s=5, label='Users')
    plt.scatter(base_stations_pso[:, 0], height - base_stations_pso[:, 1], c='green', label='PSO Base Stations', s=30)
    plt.legend()
    plt.title("PSO Base Station Distribution")

    plt.tight_layout()
    plt.show()


# ========== 曲线平滑函数 ==========
def smooth_curve(data, num_points=300):
    """平滑曲线"""
    x = np.arange(len(data))
    f = interp1d(x, data, kind='cubic')
    x_smooth = np.linspace(0, len(data) - 1, num_points)
    y_smooth = f(x_smooth)
    return x_smooth, y_smooth

# ========== 绘制 SA 覆盖率曲线 ==========
def plot_sa_coverage(sa_coverage):
    plt.figure(figsize=(12, 6))

    # 平滑曲线
    x_smooth, y_smooth = smooth_curve(sa_coverage)

    plt.plot(x_smooth, y_smooth, label="SA", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Coverage Rate")
    plt.title("SA Coverage Rate vs. Iteration")

    plt.ylim(0.01, 0.1)  # ✅ 限制纵坐标范围
    plt.grid(True)
    plt.legend()
    plt.show()

# ========== 绘制 PSO 覆盖率曲线 ==========
def plot_pso_coverage(pso_coverage):
    plt.figure(figsize=(12, 6))

    # 平滑曲线
    x_smooth, y_smooth = smooth_curve(pso_coverage)

    plt.plot(x_smooth, y_smooth, label="PSO", color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Coverage Rate")
    plt.title("PSO Coverage Rate vs. Iteration")

    plt.ylim(0.01, 0.1)  # ✅ 限制纵坐标范围
    plt.grid(True)
    plt.legend()
    plt.show()


# ========== 主程序 ==========
if __name__ == "__main__":
    image_path = "renkoumidu.jpg"
    users, width, height, weights = generate_users_exclude_white(image_path)
    # base_stations_sa = sa_optimize(users, width, height, weights)
    # base_stations_pso = pso_optimize(users, width, height, weights)
    base_stations_sa, sa_coverage_history = sa_optimize(users, width, height, weights)
    base_stations_pso, pso_coverage_history = pso_optimize(users, width, height, weights)
    plot_results(users, base_stations_sa, base_stations_pso, image_path, width, height)
    # 分开绘制 SA 和 PSO 覆盖率变化曲线
    plot_sa_coverage(sa_coverage_history)
    plot_pso_coverage(pso_coverage_history)


# Step 3: Filter out unwanted noise frequencies
filtered_ft_data= np.zeros((2, num_freqs))
filtered_ft_data[0] = ft_data[0].copy()
filtered_ft_data[1] = ft_data[1].copy()
filtered_ft_data[0][(frequencies >= 0) & (frequencies <= 890)] = 0
filtered_ft_data[1][(frequencies >= 0) & (frequencies <=890)] = 0
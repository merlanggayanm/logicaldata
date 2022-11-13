# FINAL PROJECT by Logicaldata
# Churn Prediction Analysis
#### Contributors : `Dwi Ayu S, Fadhli Mahindra, Maghfira R, M. Erlangga, Nadia N., Yohan N., Jun Kevin, M. Bara`

<br>


## Data Requirements
---
Download dataset [Disini](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

### Data Feature Explanation
---
| **Data** | **Variable** | **Description** |
|:---:|:---:|:---:|
| E Comm | CustomerID | Unique customer ID |
| E Comm | Churn | Churn Flag |
| E Comm | Tenure | Tenure of customer in organization |
| E Comm | PreferredLoginDevice | Preferred login device of customer |
| E Comm | CityTier | City tier |
| E Comm | WarehouseToHome | Distance in between warehouse to home of customer |
| E Comm | PreferredPaymentMode | Preferred payment method of customer |
| E Comm | Gender | Gender of customer |
| E Comm | HourSpendOnApp | Number of hours spend on mobile application or website |
| E Comm | NumberOfDeviceRegistered | Total number of deceives is registered on particular customer |
| E Comm | PreferedOrderCat | Preferred order category of customer in last month |
| E Comm | SatisfactionScore | Satisfactory score of customer on service |
| E Comm | MaritalStatus | Marital status of customer |
| E Comm | NumberOfAddress | Total number of added added on particular customer |
| E Comm | Complain | Any complaint has been raised in last month |
| E Comm | OrderAmountHikeFromlastYear | Percentage increases in order from last year |
| E Comm | CouponUsed | Total number of coupon has been used in last month |
| E Comm | OrderCount | Total number of orders has been places in last month |
| E Comm | DaySinceLastOrder | Day Since last order by customer |
| E Comm | CashbackAmount | Average cashback in last month |

### Data Overview
---
> ```df.head(5)```

| CustomerID | Churn | Tenure | PreferredLoginDevice | CityTier | WarehouseToHome | PreferredPaymentMode | Gender | HourSpendOnApp | NumberOfDeviceRegistered | PreferedOrderCat   | SatisfactionScore | MaritalStatus | NumberOfAddress | Complain | OrderAmountHikeFromlastYear | CouponUsed | OrderCount | DaySinceLastOrder | CashbackAmount |
|------------|-------|--------|----------------------|----------|-----------------|----------------------|--------|----------------|--------------------------|--------------------|-------------------|---------------|-----------------|----------|-----------------------------|------------|------------|-------------------|----------------|
| 50001      | 1     | 4      | Mobile Phone         | 3        | 6               | Debit Card           | Female | 3              | 3                        | Laptop & Accessory | 2                 | Single        | 9               | 1        | 11                          | 1          | 1          | 5                 | 160            |
| 50002      | 1     | NaN    | Phone                | 1        | 8               | UPI                  | Male   | 3              | 4                        | Mobile             | 3                 | Single        | 7               | 1        | 15                          | 0          | 1          | 0                 | 121            |
| 50003      | 1     | NaN    | Phone                | 1        | 30              | Debit Card           | Male   | 2              | 4                        | Mobile             | 3                 | Single        | 6               | 1        | 14                          | 0          | 1          | 3                 | 120            |
| 50004      | 1     | 0      | Phone                | 3        | 15              | Debit Card           | Male   | 2              | 4                        | Laptop & Accessory | 5                 | Single        | 8               | 0        | 23                          | 0          | 1          | 3                 | 134            |
| 50005      | 1     | 0      | Phone                | 1        | 12              | CC                   | Male   |                | 3                        | Mobile             | 5                 | Single        | 3               | 0        | 11                          | 1          | 1          | 3                 | 130            |


## Machine Learning Workflow
![Google Drive Image](https://drive.google.com/uc?export=view&id=1oHFeic_FP2p7W0Ffh_dce0xfAsXHup5N)
Dalam langkah Churn Prediction Analysis terbagi dalam beberapa Stage
1. Stage 0: Problem Statements, Goal, & Objective Analysis
2. Stage 1: EDA, Insights & Visialization (Data Understanding)
3. Stage 2: Data Processing (Data Cleansing & Feature Engineering)
4. Stage 3: Machine Learning Modelling & Model Evaluation

## Stage 0 :
### Problem Statements
Shoptify merupakan sebuah e-commerce yang saat ini mengalami customer churn sebesar 16.8%. Hal tersebut menyebabkan Shoptify kehilangan banyak customer sehingga akan berdampak kepada revenue Shoptify. Selain itu, biaya untuk mengakuisisi customer baru jauh lebih besar daripada biaya untuk mempertahankan customer lama. Oleh karena itu, Shoptify perlu melakukan identifikasi penyebab customer churn dan solusi untuk mengatasi permasalahan tersebut. Dengan memprediksi customer mana saja yang memiliki kemungkinan untuk churn, Shoptify dapat menentukan langkah strategi marketing yang tepat sesuai dengan karakteristik customer.

### Goal
Memprediksi tipe pelanggan yang sangat berpotensial melakukan churn dengan melakukan identifikasi fitur untuk meminimalisir tingkat churn pelanggan dan mendapatkan keputusan bisnis yang tepat.

### Objective
1. Membuat model Machine Learning untuk memprediksi user yang berpotensi churn serta memahami fitur apa saja yang menyebabkan user menjadi churn dan fitur yang memacu potensi meminimalisir tingkat churn
2. Mengoptimalkan revenue perusahaan dengan mengelompokan user yang berpotensi churn agar dapat diberikan treatment berbeda sehingga churn rate menurun.

### Business Metric
1. Customer churn rate yang merupakan presentase customer yang berhenti menggunakan e-commerce.
2. Revenue yang merupakan metric utama dalam bisnis e-commerce.

## Stage 1 :
### EDA & Insight Summary
1. Terdapat 7 variabel yang memiliki missing values.
2. Rata-rata pelanggan menghabiskan waktu selama 3 jam dalam menggunakan aplikasi e-commerce.
3. Masih banyak pelanggan yang tidak menggunakan kupon dalam berbelanja.
4. Pelanggan dengan lokasi lebih dekat dengan gudang (< 15 km) labih mendominasi.
5. Jumlah pelanggan yang melakukan komplain cukup banyak sekitar 1000 orang. Hal ini harus menjadi perhatian dalam perkembangan bisnis ke depannya.
6. Variabel Tenure menjadi perhatian karena banyaknya pelanggan churn pada tenure < 2 bulan.
7. Variabel Tenure & Complain memiliki korelasi yang cukup baik dengan variable target (Churn) yaitu -0,35 dan 0,25.
8. Tingkat customer yang churn lebih banyak pada pelanggan yang menggunakan sedikit kupon (1-2 saja) bahkan oleh pelanggan yang tidak menggunakan kupon sama sekali. Keputusan bisnis yang bisa dilakukan adalah memberikan coupon setiap tiap weekend atau tanggal tertentu.
9. Tidak ada jaminan bahwa jika satisfaction score tinggi, maka probabilitas churn semakin rendah.

## Stage 2 :
### Preprocessing
1. Dari keseluruhan data masih terdapat perbedaan pengisian data pada kategori tertentu, sehingga pada data tersebut dilakukan replace terlebih dahulu ke dalam satu value agar menghilangkan redudansi, yakni pada kolom PreferredLoginDevice dan PreferredPaymentMode dengan mereplace Phone menjadi Mobile Phone, kemudian CC menjadi Credit Card, lalu Cash On Delivery menjadi COD. Untuk menangani value feature yang masih kosong dilakukan imputasi dengan menginput nilai median dari setiap featue dikarenakan nilai mean dan median yang tidak timpang jauh dan nilai median bernilai absolut.
2. Setelah dilakukakn pengecekan dataset tidak terdapat baris data yang memiliki nilai sama pada semua featurenya.
3. Dengan metode z score, outliers yang akan dihilangkan sekitar 0,2 % yaitu 11 baris data yang akan di drop.
4. Data categorical yang bertipe data object/string perlu diubah menjadi tipe data numerikal dengan feature encoding yang terbagi menjadi dua metode, untuk feature gender dilakukan label encoding dan untuk data yang memiliki lebih dari 2 kategori dilakukan one hot encoding.
5. Tahap awal pada feature transformation yaitu menklasifikasi apakah feature tersebut memiliki distribusi normal atau tidak berdasarkan nilai skew dan kurtosis. Setelah itu, dilakukan split dari data keseluruhan menjadi train set sebesar 70% dan test set sebesar 30%. Lalu, digunakan standardization untuk menyesuaikan nilai antar feature. Data training kemudian di-scaling menggunakan Standar Scaler kemudian dilakukan fit & transform, sedangkan pada data testing hanya dilakukan transform.
6. Feature yang kurang relevan dalam proses pemodelan akan di drop, yaitu feature Customer ID sehingga total feature yang digunakan sebanyak 5 numerical feature, 13 categorical fitur dan 1 fitur target yaitu 'Churn'.
7. Pada fitur target (Churn) terdapat imbalance data yang signifikan yaitu dengan nilai false sebanyak 83,12% dan nilai true sebesar 16,84%.Untuk menangani kondisi tersebut dilakukan proses oversampling untuk menyamakan rasio pada data training saja dengan menggunakan metode SMOTE yaitu dengan generate value baru untuk variablenya, berdasarkan distribusi data tersebut agar rasio data seimbang.

<h1>Implementasi Algoritma Random Forest dan Multinomial Naive Bayes dengan Menggunakan Seleksi Fitur Information Gain untuk Klasifikasi Berita Media Monitoring Kawasan Geopark Ciletuh</h1>

<p>Ini merupakan repository yang berisi source code dan dataset yang digunakan untuk keperluan Tugas Akhir dengan judul Implementasi Algoritma Random Forest dan Multinomial Naive Bayes dengan Menggunakan Seleksi Fitur Information Gain untuk Klasifikasi Berita Media Monitoring Kawasan Geopark Ciletuh</p>

<h2>Abstrak</h2>

<p>Pada era teknologi informasi dan komunikasi seperti saat ini, proses penyebaran informasi menjadi lebih masif dan cepat.  Hal ini menyebabkan proses media <em>monitoring</em> yang dilakukan praktisi <em>Public Relations</em> (PR) untuk mengembangkan dan mempertahankan reputasi kawasan <em>Geopark</em> Ciletuh menjadi kurang efektif dan efisien apabila dalam melakukan identifikasi dan analisis berita masih dilakukan secara manual. Dalam sistem media <em>monitoring</em>, proses ini masuk kedalam proses <em>Analysis Backend</em>. Penelitian ini mengusulkan sistem klasifikasi berita dengan menggunakan algoritma <em>machine learning</em> untuk membuat proses analisis berita yang dilakukan lebih efektif dan efisi-en dengan menggunakan dua algoritma <em>Machine Learning</em> yaitu <em>Random Forest</em> dan <em>Multinomial Naive Bayes</em> serta <em>Information Gain</em> sebagai metode pemilihan fitur dengan nilai <em>threshold</em> 0.05 dan 0.01 untuk melakukan klasifikasi berita <em>hard news</em> atau <em>soft news</em>. Dataset dibentuk berdasarkan dua model yang berbeda yaitu <em>single dimensional</em> dan <em>multidimensional</em>. Hasil penelitian untuk model dataset <em>single dimensional</em> dengan algoritma <em>Random Forest</em> memperoleh rata-rata tertinggi untuk nilai akurasi sebesar 81.42% menggunakan pemilihan fitur <em>information gain</em> dengan <em>threshold</em> 0.01, sedangkan algoritma <em>Multinomial Naive Bayes</em> memperoleh rata-rata tertinggi untuk nilai akurasi sebesar 74.18% menggunakan <em>information gain</em> dengan <em>threshold</em> 0.01 dan tanpa pemilihan fitur. Untuk model dataset <em>multidimensional</em> algoritma <em>Random Forest</em> memperoleh  rata-rata tertinggi untuk nilai akurasi 93.8%, sedangkan algoritma <em>Multinomial Naive Bayes</em> sebesar 72.72%. Dari hasil yang diperoleh, penggunaan <em>information gain</em> untuk pemilihan fitur memberikan performa yang kurang baik untuk algoritma <em>Multinomial Naive Bayes</em> karena cara kerja algoritma yang menganggap semua fitur bersifat independen dan penggunaan metode <em>laplacian smoothing</em> membuatpengurangan fitur yang dilakukan tidak memberikan hasil yang baik.</p>

<h2>Bahasa Pemrograman dan <em>Library</em></h2>
<ul>
  <li>Python 3.7</li>
  <li>Scikit-Learn</li>
  <li>NumPy</li>
  <li>Natural Language Tool Kit (NLTK)</li>
  <li>Sastrawi Stemming Python</li>
  <li>Pandas DataFrame</li>
</ul>

<h2>Penggunaan</h2>
<ol>
  <li>Dataset Preparation</li>
  <li>Seleksi/pemilihan fitur yang akan digunakan</li>
  <li>Melakukan klasifikasi/prediksi</li>
</ol>

<h3>Dataset</h3>
<p>Berikut dataset yang digunakan dalam penelitian ini, diantaranya:</p>
<ul>
  <li>Dataset Single Dimensional (single-dataset.csv)</li>
  <li>Dataset Single Dimensional dengan Information Gain 0.01 (Dt-IG(0.01).csv)</li>
  <li>Dataset Single Dimensional dengan Information Gain 0.05 (Dt-IG(0.05).csv)</li>
  <li>Dataset Single Dimensional dengan Chi Square 0.01 (chiSqr(0.01).csv)</li>
  <li>Dataset Single Dimensional dengan Chi Square 0.05 (chiSqr(0.05).csv)</li>
  <li>Dataset Multidimensional (multidimensional-dataset.csv)</li>
</ul>

<h3>1. Dataset Preparation</h3>
<p>Untuk tahapan preprocessing, penentuan fitur pembentuk dataset dengan bag-of-words, dan pembobotan atribut/fitur dengan TF-IDF untuk membentuk dataset single dimensional dilakukan dengan menjalankan script pada single_dataset_prep.py menggunakan input raw.csv.</p>

<h3>2. Seleksi/Pemilihan Fitur</h3>
<p>Dalam penelitian ini metode seleksi/pemilihan fitur Information Gain digunakan untuk mengurangi fitur-fitur yang kurang relevan bagi proses klasifikasi berita yang akan dilakukan. Untuk dapat melakukan seleksi/pemilihan fitur dengan Information Gain dapat menjalankan script pada fitur_info_gain.py dengan menggunakan input data dari file single-dataset.csv. Selain itu, dalam penelitian ini juga menggunakan satu tambahan pemilihan fitur yang digunakan sebagai pembanding performa dari Information Gain terhadap metode pemilihan fitur lain yaitu Chi-Square yang dapat dijalankan melalui script fitur_chisqr.py dengan menggunakan input data yang sama yaitu single-dataset.csv</p>

<h3>3. Klasifikasi Berita Hard/Soft News</h3>
<p>Dalam penelitian ini pembentukan model menggunakan dua algoritma machine learning untuk melakukan klasifikasi berita Hard/Soft News terkait Geopark Ciletuh yaitu algoritma Random Forest dan algoritma Multinomial Naive Bayes. Untuk menjalankan model dengan algoritma Random Forest dapat melalui script model_random_forest.py, dan untuk menjalankan model dengan algoritma Multinomial Naive Bayes dapat melalui script model_mnb.py</p>

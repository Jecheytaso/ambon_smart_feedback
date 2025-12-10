document.getElementById("btnDeteksi").addEventListener("click", async () => {
  const teks = document.getElementById("inputTeks").value.trim();
  const hasilBox = document.getElementById("hasilDeteksi");
  const outputKelas = document.getElementById("outputKelas");

  if (!teks) {
    alert("Masukkan teks aduan terlebih dahulu!");
    return;
  }

  hasilBox.style.display = "block";
  outputKelas.innerHTML = "⏳ Sedang menganalisis...";

  try {
    const response = await fetch("/predict", {  
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: teks })
    });

    if (!response.ok) throw new Error("Gagal memuat data dari server");

    const data = await response.json();
    const hasil = data.prediction || "Tidak diketahui";

    const warna = {
      "Dinas Kominfo": "#007BFF",
      "Dinas Lingkungan Hidup dan Persampahan": "#28A745",
      "Dinas Pekerjaan Umum dan Penataan Ruangan": "#FFC107",
      "Dinas Pendidikan": "#17A2B8",
      "Dinas Perhubungan": "#DC3545"
    };

    outputKelas.innerHTML = `<span style="color:${warna[hasil] || '#333'}; font-weight:bold;">🏛️ ${hasil}</span>`;
  } catch (error) {
    console.error(error);
    outputKelas.innerHTML = "❌ Terjadi kesalahan: tidak bisa terhubung ke server.";
  }
});
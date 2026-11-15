// Live character counter for the tweet box.
const box = document.getElementById("tweet");
const count = document.getElementById("count");

if (box && count) {
  const update = () => { count.textContent = box.value.length; };
  box.addEventListener("input", update);
  update();
}

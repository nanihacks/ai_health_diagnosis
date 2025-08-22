document.addEventListener('DOMContentLoaded', () => {
  // Simple form validation with smooth error highlighting
  const forms = document.querySelectorAll('form');

  forms.forEach(form => {
    form.addEventListener('submit', event => {
      let valid = true;
      const requiredFields = form.querySelectorAll('input[required], select[required], textarea[required]');
      requiredFields.forEach(field => {
        if (!field.value.trim()) {
          valid = false;
          field.classList.add('input-error');
          // Shake animation to draw attention
          field.style.animation = 'shake 0.4s ease';
          field.addEventListener('animationend', () => field.style.animation = '');
        } else {
          field.classList.remove('input-error');
        }
      });
      if (!valid) {
        event.preventDefault();
        alert('Please fill in all required fields.');
      }
    });
  });
});

// Shake animation for error input feedback
const style = document.createElement('style');
style.innerHTML = `
@keyframes shake {
  0%, 100% { transform: translateX(0); }
  25% { transform: translateX(-6px);}
  50% { transform: translateX(6px);}
  75% { transform: translateX(-6px);}
}
`;
document.head.appendChild(style);

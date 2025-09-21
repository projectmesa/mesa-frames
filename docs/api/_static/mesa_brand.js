// Small script to add a theme toggle to the navbar and integrate with pydata theme
(function(){
  function createToggle(){
    try{
      var btn = document.createElement('button');
      btn.className = 'theme-switch-button btn btn-sm';
      btn.type = 'button';
      btn.title = 'Toggle theme';
      btn.setAttribute('aria-label','Toggle theme');
      btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 3v2M12 19v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4M12 7a5 5 0 100 10 5 5 0 000-10z" stroke="#fff" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      var container = document.querySelector('.navbar-icon-links') || document.querySelector('.bd-navbar-elements') || document.querySelector('.navbar .navbar-nav') || document.querySelector('.pydata-navbar .navbar-nav');
      if(container){
        var li = document.createElement('li');
        li.className = 'nav-item';
        var a = document.createElement('a');
        a.className = 'nav-link';
        a.href = '#';
        a.appendChild(btn);
        li.appendChild(a);
        // insert at the end of the list so we don't disrupt other items
        container.appendChild(li);

        btn.addEventListener('click', function(e){
          e.preventDefault();
          // Try to reuse pydata theme switch if available
          try{
            // cycleMode function may be defined by pydata theme; call if present
            if(typeof cycleMode === 'function'){
              cycleMode();
              return;
            }
            // fallback: toggle data-mode between dark and light and persist
            var current = document.documentElement.getAttribute('data-mode') || '';
            var next = (current === 'dark') ? 'light' : 'dark';
            document.documentElement.setAttribute('data-mode', next);
            document.documentElement.dataset.mode = next;
            try{ localStorage.setItem('mode', next); }catch(e){}
          }catch(err){ console.warn('Theme toggle failed', err);}        
        });
      }
    }catch(e){console.warn('mesa_brand.js init fail',e);}  }
  document.addEventListener('DOMContentLoaded', createToggle);
})();

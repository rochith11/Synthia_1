(() => {
  const prefersReducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  const closeMobileMenu = (nav, toggle) => {
    nav.dataset.open = "false";
    toggle.setAttribute("aria-expanded", "false");
  };

  const initMenu = () => {
    const toggle = document.querySelector(".menu-toggle");
    const nav = document.querySelector(".site-nav");
    if (!toggle || !nav) {
      return;
    }

    toggle.addEventListener("click", () => {
      const isOpen = nav.dataset.open === "true";
      nav.dataset.open = String(!isOpen);
      toggle.setAttribute("aria-expanded", String(!isOpen));
    });

    nav.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", () => {
        closeMobileMenu(nav, toggle);
      });
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 840) {
        closeMobileMenu(nav, toggle);
      }
    });
  };

  const initRevealAnimations = () => {
    const nodes = [...document.querySelectorAll("[data-animate]")];
    if (!nodes.length) {
      return;
    }

    if (prefersReducedMotion || !("IntersectionObserver" in window)) {
      nodes.forEach((node) => node.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.18,
        rootMargin: "0px 0px -8% 0px",
      }
    );

    nodes.forEach((node, index) => {
      node.style.transitionDelay = `${Math.min(index * 45, 220)}ms`;
      observer.observe(node);
    });
  };

  const parseNumeric = (rawValue) => {
    const cleaned = String(rawValue).replace(/,/g, "").trim();
    const value = Number.parseFloat(cleaned);

    if (Number.isNaN(value)) {
      return null;
    }

    const fractionalPart = cleaned.includes(".")
      ? cleaned.split(".")[1].length
      : 0;

    return {
      value,
      decimals: Math.min(fractionalPart, 6),
    };
  };

  const formatNumber = (value, decimals) =>
    new Intl.NumberFormat("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }).format(value);

  const initCountUp = () => {
    const nodes = [...document.querySelectorAll("[data-countup]")];
    if (!nodes.length) {
      return;
    }

    const animateNode = (node) => {
      const parsed = parseNumeric(node.dataset.countup);
      if (!parsed) {
        return;
      }

      if (prefersReducedMotion) {
        node.textContent = formatNumber(parsed.value, parsed.decimals);
        return;
      }

      const duration = 900;
      const startedAt = performance.now();

      const tick = (now) => {
        const elapsed = Math.min((now - startedAt) / duration, 1);
        const eased = 1 - (1 - elapsed) * (1 - elapsed);
        const current = parsed.value * eased;

        node.textContent = formatNumber(current, parsed.decimals);

        if (elapsed < 1) {
          requestAnimationFrame(tick);
        } else {
          node.textContent = formatNumber(parsed.value, parsed.decimals);
        }
      };

      requestAnimationFrame(tick);
    };

    if (!("IntersectionObserver" in window) || prefersReducedMotion) {
      nodes.forEach(animateNode);
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animateNode(entry.target);
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );

    nodes.forEach((node) => observer.observe(node));
  };

  const initLoadingForms = () => {
    document.querySelectorAll(".js-loading-form").forEach((form) => {
      form.addEventListener("submit", () => {
        const submitButton = form.querySelector('button[type="submit"]');
        if (!submitButton || submitButton.disabled) {
          return;
        }

        submitButton.textContent =
          submitButton.dataset.loadingText || "Working...";
        submitButton.classList.add("is-loading");
        submitButton.disabled = true;
      });
    });
  };

  const initFlashTimeout = () => {
    const flashes = [...document.querySelectorAll(".flash")];
    if (!flashes.length) {
      return;
    }

    window.setTimeout(() => {
      flashes.forEach((flash, index) => {
        window.setTimeout(() => flash.classList.add("flash-hide"), index * 110);
      });
    }, 4200);
  };

  document.addEventListener("DOMContentLoaded", () => {
    initMenu();
    initRevealAnimations();
    initCountUp();
    initLoadingForms();
    initFlashTimeout();
  });
})();

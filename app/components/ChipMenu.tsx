"use client";

import { ChevronDown } from "lucide-react";
import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

export const chipClass =
  "inline-flex shrink-0 items-center gap-1 rounded-full border border-slate-200/80 bg-white/70 px-2.5 py-1 text-[12.5px] font-medium text-slate-700 transition-colors hover:bg-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-electric-light disabled:opacity-50";

export interface MenuItem<T extends string> {
  value: T;
  label: string;
  description?: string;
  thumbnail?: string;
}

interface Props<T extends string> {
  label: string;
  ariaLabel: string;
  icon: ReactNode;
  items: MenuItem<T>[];
  value: T | null;
  onSelect: (value: T) => void;
  width?: number;
}

export function ChipMenu<T extends string>({ label, ariaLabel, icon, items, value, onSelect, width = 240 }: Props<T>) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ left: 0, top: 0 });
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  function toggle() {
    if (open) return setOpen(false);
    const r = buttonRef.current!.getBoundingClientRect();
    setPos({ left: Math.max(8, Math.min(r.left, window.innerWidth - width - 8)), top: r.bottom + 8 });
    setOpen(true);
  }

  function close(returnFocus: boolean) {
    setOpen(false);
    if (returnFocus) buttonRef.current?.focus();
  }

  // flip above the chip when there is no room below
  useLayoutEffect(() => {
    if (!open || !menuRef.current || !buttonRef.current) return;
    const h = menuRef.current.offsetHeight;
    const r = buttonRef.current.getBoundingClientRect();
    if (r.bottom + 8 + h > window.innerHeight && r.top - 8 - h > 0) {
      menuRef.current.style.top = `${r.top - 8 - h}px`;
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const menu = menuRef.current;
    const target = menu?.querySelector<HTMLElement>('[aria-checked="true"]') ?? menu?.querySelector<HTMLElement>('[role="menuitemradio"]');
    target?.focus({ preventScroll: true });

    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!menu?.contains(t) && !buttonRef.current?.contains(t)) setOpen(false);
    };
    const onScroll = (e: Event) => {
      if (!menu?.contains(e.target as Node)) setOpen(false);
    };
    const onResize = () => setOpen(false);
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [open]);

  function onKeyDown(e: React.KeyboardEvent) {
    const options = Array.from(menuRef.current?.querySelectorAll<HTMLElement>('[role="menuitemradio"]') ?? []);
    const i = options.indexOf(document.activeElement as HTMLElement);
    const focus = (j: number) => options[(j + options.length) % options.length]?.focus();
    if (e.key === "ArrowDown") focus(i + 1);
    else if (e.key === "ArrowUp") focus(i - 1);
    else if (e.key === "Home") focus(0);
    else if (e.key === "End") focus(options.length - 1);
    else if (e.key === "Escape") close(true);
    else if (e.key === "Tab") return close(false);
    else return;
    e.preventDefault();
  }

  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className={chipClass}
        aria-label={`${ariaLabel}: ${label}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={toggle}
      >
        {icon}
        <span>{label}</span>
        <ChevronDown className="size-3.5 text-slate-400" aria-hidden />
      </button>
      {open &&
        createPortal(
          <div
            ref={menuRef}
            id={menuId}
            role="menu"
            aria-label={ariaLabel}
            onKeyDown={onKeyDown}
            style={{ left: pos.left, top: pos.top, width }}
            className="fixed z-50 max-h-[70vh] overflow-y-auto rounded-2xl border border-slate-200/80 bg-white/95 p-1.5 text-left shadow-[0_16px_40px_-12px_rgba(10,40,120,0.35)] backdrop-blur-xl"
          >
            {items.map((item) => (
              <button
                key={item.value}
                type="button"
                role="menuitemradio"
                aria-checked={item.value === value}
                onClick={() => {
                  onSelect(item.value);
                  close(true);
                }}
                className="flex w-full items-center gap-3 rounded-xl px-2.5 py-2 text-left text-sm text-slate-700 outline-none hover:bg-mist focus-visible:bg-mist aria-checked:bg-mist/70"
              >
                {item.thumbnail && (
                  // eslint-disable-next-line @next/next/no-img-element -- tiny static thumbnails inside a portal menu
                  <img src={item.thumbnail} alt="" className="size-11 shrink-0 rounded-lg border border-slate-200 bg-white object-cover" />
                )}
                <span className="min-w-0 flex-1">
                  <span className="block font-medium text-ink">{item.label}</span>
                  {item.description && <span className="block truncate text-xs text-slate-500">{item.description}</span>}
                </span>
                {item.value === value && <span className="size-1.5 shrink-0 rounded-full bg-electric-light" aria-hidden />}
              </button>
            ))}
          </div>,
          document.body,
        )}
    </>
  );
}

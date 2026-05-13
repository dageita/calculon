import { useRef, useCallback, useEffect, type KeyboardEvent } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';

/** 实心发送图标：使用显式 fill，避免 currentColor 在 antd button 样式下不可见 */
function SendArrowIcon({ active }: { active: boolean }) {
  const fill = active ? '#fafafa' : '#475569';
  return (
    <svg
      width={20}
      height={20}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
      focusable="false"
      style={{ display: 'block', flexShrink: 0 }}
    >
      <path
        fill={fill}
        d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z"
      />
    </svg>
  );
}

interface UseAutoResizeTextareaProps {
  minHeight: number;
  maxHeight?: number;
}

function useAutoResizeTextarea({ minHeight, maxHeight }: UseAutoResizeTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      if (reset) {
        textarea.style.height = `${minHeight}px`;
        return;
      }

      textarea.style.height = `${minHeight}px`;

      const newHeight = Math.max(
        minHeight,
        Math.min(textarea.scrollHeight, maxHeight ?? Number.POSITIVE_INFINITY),
      );

      textarea.style.height = `${newHeight}px`;
    },
    [minHeight, maxHeight],
  );

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = `${minHeight}px`;
    }
  }, [minHeight]);

  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [adjustHeight]);

  return { textareaRef, adjustHeight };
}

export interface AI_PromptProps {
  value: string;
  onValueChange: (value: string) => void;
  onSubmit: () => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export function AI_Prompt({
  value,
  onValueChange,
  onSubmit,
  disabled,
  placeholder = '在此输入消息…',
  className,
}: AI_PromptProps) {
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 104,
    maxHeight: 380,
  });

  useEffect(() => {
    if (value === '') {
      adjustHeight(true);
    }
  }, [value, adjustHeight]);

  const trySubmit = () => {
    if (disabled || !value.trim()) return;
    onSubmit();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && value.trim() && !disabled) {
      e.preventDefault();
      trySubmit();
    }
  };

  const canSend = Boolean(value.trim()) && !disabled;

  return (
    <div className={cn('w-full min-w-0', className)}>
      {/* 外层与 Agent 深色面板衔接；内层为可输入的「白框」 */}
      <div className="rounded-xl border border-white/[0.12] bg-[#141418] p-1.5 sm:p-2">
        <div
          className="agent-sim-input-white relative w-full min-w-0 rounded-lg border border-zinc-300/90 bg-white shadow-[inset_0_1px_0_rgba(255,255,255,0.9)]"
          style={{ position: 'relative' }}
        >
          <Textarea
            id="ai-input-bar"
            value={value}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            cols={1}
            style={{ width: '100%', resize: 'none', color: '#0f172a' }}
            className={cn(
              'agent-sim-ai-textarea box-border min-h-[104px] w-full max-w-full overflow-y-auto rounded-lg border-0 bg-transparent',
              'px-3 pb-12 pt-2.5 pr-14 text-[15px] leading-relaxed shadow-none sm:pr-16',
              'placeholder:text-zinc-400',
              'focus-visible:ring-0 focus-visible:ring-offset-0',
            )}
            ref={textareaRef}
            onKeyDown={handleKeyDown}
            onChange={(e) => {
              onValueChange(e.target.value);
              adjustHeight();
            }}
          />

          <button
            type="button"
            className={cn(
              'agent-sim-send-btn flex h-9 w-9 items-center justify-center rounded-full border border-zinc-700/80 bg-zinc-900 text-white shadow-sm transition-colors duration-150 hover:border-teal-500/50 hover:bg-zinc-800 active:bg-black sm:h-10 sm:w-10',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/70 focus-visible:ring-offset-2 focus-visible:ring-offset-white',
              !canSend &&
                'cursor-not-allowed border-zinc-200 bg-zinc-100 text-zinc-400 hover:border-zinc-200 hover:bg-zinc-100 active:bg-zinc-100',
            )}
            style={{
              position: 'absolute',
              right: '10px',
              bottom: '10px',
              zIndex: 30,
            }}
            aria-label="发送"
            disabled={!canSend}
            onClick={trySubmit}
          >
            <SendArrowIcon active={canSend} />
          </button>
        </div>
      </div>
    </div>
  );
}

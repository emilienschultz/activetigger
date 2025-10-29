import { FC, useCallback, useEffect, useState } from 'react';
import { IoMdSkipForward } from 'react-icons/io';
import { AppContextValue } from '../core/context';

// This component is used to skip the current element in the annotation session

interface ForwardButtonProps {
  elementId: string;
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
  refetchElement: () => void;
}

export const ForwardButton: FC<ForwardButtonProps> = ({
  setAppContext,
  elementId,
  refetchElement,
}) => {
  const [shouldRefetch, setShouldRefetch] = useState(false);

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      // prevent shortkey to perturb the inputs
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      if (ev.code === 'KeyP') {
        setAppContext((prev) => ({ ...prev, history: [...prev.history, elementId] }));
        setShouldRefetch(true);
      }
    },
    [elementId, setShouldRefetch, setAppContext],
  );

  useEffect(() => {
    // manage keyboard shortcut if less than 10 label
    document.addEventListener('keydown', handleKeyboardEvents);

    return () => {
      document.removeEventListener('keydown', handleKeyboardEvents);
    };
  }, [handleKeyboardEvents]);

  useEffect(() => {
    if (shouldRefetch) {
      refetchElement();
      setShouldRefetch(false);
    }
  }, [shouldRefetch, refetchElement]);

  return (
    <button
      className="btn nextelement"
      onClick={() => {
        setAppContext((prev) => ({ ...prev, history: [...prev.history, elementId] }));
        setShouldRefetch(true);
      }}
    >
      <IoMdSkipForward title="Skip" />
      {/* <Tooltip anchorSelect=".nextelement" place="top">
        Skip
      </Tooltip> */}
    </button>
  );
};

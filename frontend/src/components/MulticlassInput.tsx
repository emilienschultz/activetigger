import { FC, useCallback, useEffect, useState } from 'react';
import { ReactSortable } from 'react-sortablejs';

interface MulticlassInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string) => void;
}

interface LabelType {
  id: number;
  label: string;
}

export const MulticlassInput: FC<MulticlassInputProps> = ({
  elementId,
  postAnnotation,
  labels,
}) => {
  const [availableLabels, setAvailableLabels] = useState<LabelType[]>(
    (labels || []).map((label, index) => ({
      id: index,
      label: label,
    })),
  );

  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      // prevent shortkey to perturb the inputs
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      availableLabels.forEach((item, i) => {
        if (ev.code === `Digit` + (i + 1) || ev.code === `Numpad` + (i + 1)) {
          postAnnotation(item.label, elementId);
        }
      });
    },
    [availableLabels, postAnnotation, elementId],
  );

  useEffect(() => {
    // manage keyboard shortcut if less than 10 label
    if (availableLabels.length > 0 && availableLabels.length < 10) {
      document.addEventListener('keydown', handleKeyboardEvents);
    }

    return () => {
      if (availableLabels.length > 0 && availableLabels.length < 10) {
        document.removeEventListener('keydown', handleKeyboardEvents);
      }
    };
  }, [availableLabels, handleKeyboardEvents]);

  return (
    <ReactSortable list={availableLabels} setList={setAvailableLabels} tag="div">
      {
        // display buttons for label from the user
        availableLabels.map((e, i) => (
          <button
            type="button"
            key={e.label}
            value={e.label}
            className="btn btn-primary grow-1 gap-2 justify-content-center m-1"
            onClick={(v) => {
              postAnnotation(v.currentTarget.value, elementId);
            }}
          >
            {e.label} <span className="badge text-bg-secondary">{i + 1}</span>
          </button>
        ))
      }
    </ReactSortable>
  );
};

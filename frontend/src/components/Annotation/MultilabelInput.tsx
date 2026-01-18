import { FC, useCallback, useEffect, useState } from 'react';
import { FaCheck } from 'react-icons/fa6';
import Select from 'react-select';

interface MulticlassInputProps {
  elementId: string;
  labels: string[];
  postAnnotation: (label: string, elementId: string) => void;
}

export const MultilabelInput: FC<MulticlassInputProps> = ({
  elementId,
  postAnnotation,
  labels,
}) => {
  // management multilabels
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);

  // add shortcut
  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      if (ev.key === 'Enter' && ev.ctrlKey) {
        postAnnotation(selectedLabels.join('|'), elementId);
        setSelectedLabels([]);
      }
    },
    [postAnnotation, selectedLabels, elementId],
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyboardEvents);
    return () => {
      document.removeEventListener('keydown', handleKeyboardEvents);
    };
  }, [handleKeyboardEvents]);

  return (
    <div className="d-flex gap-2 align-items-center">
      <Select
        isMulti
        options={labels.map((e) => ({ value: e, label: e }))}
        onChange={(e) => {
          setSelectedLabels(e.map((e) => e.value));
        }}
        value={selectedLabels.map((e) => ({ value: e, label: e }))}
        className="w-100"
      />
      <button
        className="btn btn-outline-success d-flex align-items-center justify-content-center validate-btn"
        onClick={() => {
          postAnnotation(selectedLabels.join('|'), elementId);
          setSelectedLabels([]);
        }}
      >
        <FaCheck size={18} />
      </button>
    </div>
  );
};

import { FC, useCallback, useEffect, useState } from 'react';
import { FaSquareCheck } from 'react-icons/fa6';
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
    <div className="horizontal" style={{ width: '50%', minWidth: '300px' }}>
      {/* NOTE: Axel: Need to be reconsidered */}
      <Select
        isMulti
        className="flex-grow-1" // Axel: ugghhhhh
        options={labels.map((e) => ({ value: e, label: e }))}
        onChange={(e) => {
          setSelectedLabels(e.map((e) => e.value));
        }}
        value={selectedLabels.map((e) => ({ value: e, label: e }))}
      />
      <button
        className="btn-primary-action"
        onClick={() => {
          postAnnotation(selectedLabels.join('|'), elementId);
          setSelectedLabels([]);
        }}
      >
        <FaSquareCheck size={30} />
      </button>
    </div>
  );
};

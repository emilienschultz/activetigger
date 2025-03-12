import { FC, useCallback, useEffect } from 'react';
import { IoMdSkipBackward } from 'react-icons/io';
import { Link, useNavigate } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { AppContextValue } from '../core/context';

interface BackButtonProps {
  projectName: string;
  history: string[];
  setAppContext: React.Dispatch<React.SetStateAction<AppContextValue>>;
}

export const BackButton: FC<BackButtonProps> = ({ projectName, history, setAppContext }) => {
  const navigate = useNavigate();
  const handleKeyboardEvents = useCallback(
    (ev: KeyboardEvent) => {
      // prevent shortkey to perturb the inputs
      const activeElement = document.activeElement;
      const isFormField =
        activeElement?.tagName === 'INPUT' ||
        activeElement?.tagName === 'TEXTAREA' ||
        activeElement?.tagName === 'SELECT';
      if (isFormField) return;

      if (ev.code === 'Backspace') {
        navigate(`/projects/${projectName}/annotate/${history[history.length - 1]}`);
        setAppContext((prev) => ({ ...prev, history: prev.history.slice(0, -1) }));
      }
    },
    [setAppContext, navigate, projectName, history],
  );

  useEffect(() => {
    // manage keyboard shortcut if less than 10 label
    document.addEventListener('keydown', handleKeyboardEvents);

    return () => {
      document.removeEventListener('keydown', handleKeyboardEvents);
    };
  }, [handleKeyboardEvents]);

  return (
    <Link
      to={`/projects/${projectName}/annotate/${history[history.length - 1]}`}
      className="btn previouselement"
      onClick={() => {
        setAppContext((prev) => ({ ...prev, history: prev.history.slice(0, -1) }));
      }}
    >
      <IoMdSkipBackward />
      <Tooltip anchorSelect=".previouselement" place="top">
        Go back to previous element
      </Tooltip>
    </Link>
  );
};

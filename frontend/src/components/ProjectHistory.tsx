import { Dispatch, FC, SetStateAction } from 'react';

import { useGetLogs } from '../core/api';

import DataGrid, { Column } from 'react-data-grid';
import { HiOutlineQuestionMarkCircle } from 'react-icons/hi';
import { Tooltip } from 'react-tooltip';
import { AppContextValue } from '../core/context';

interface ProjectHistoryProps {
  projectSlug: string;
  history: string[];
  setAppContext: Dispatch<SetStateAction<AppContextValue>>;
}

interface Row {
  time: string;
  user: string;
  action: string;
}

export const ProjectHistory: FC<ProjectHistoryProps> = ({
  projectSlug,
  history,
  setAppContext,
}) => {
  // get logs
  const { logs } = useGetLogs(projectSlug, 100);

  // function to clear history
  const actionClearHistory = () => {
    setAppContext((prev) => ({ ...prev, history: [] }));
  };

  const columns: readonly Column<Row>[] = [
    {
      name: 'Time',
      key: 'time',
      resizable: true,
    },
    {
      name: 'User',
      key: 'user',
      resizable: true,
    },
    {
      name: 'Project',
      key: 'project',
    },
    {
      name: 'Action',
      key: 'action',
    },
  ];
  return (
    <div>
      <div>
        Session counter{' '}
        <a className="history">
          <HiOutlineQuestionMarkCircle />
        </a>
        <Tooltip anchorSelect=".history" place="top">
          Element annotated during this session. If you annotate already annotated data, it prevents
          you to see an element twice. Clear it if you want to be able to re-annotate again.
        </Tooltip>{' '}
        <span className="badge rounded-pill text-bg-light text-muted me-2" key={history.length}>
          {history.length}
        </span>
      </div>
      <button onClick={actionClearHistory} className="delete-button">
        Clear history
      </button>
      <div className="subsection">Activity on this project</div>
      <DataGrid<Row>
        className="fill-grid mt-2"
        columns={columns}
        rows={(logs as unknown as Row[]) || []}
      />
    </div>
  );
};

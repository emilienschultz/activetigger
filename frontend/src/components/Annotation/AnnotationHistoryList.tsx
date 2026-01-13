import classNames from 'classnames';
import { truncate } from 'lodash';
import { FC } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { Link, useParams } from 'react-router-dom';
import { useAppContext } from '../../core/context';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { ElementHistoryPoint } from '../../types';
import { AnnotationIcon, NoAnnotationIcon } from '../Icons';
import { MiddleEllipsis } from './MiddleEllipsis';

const dateTimeFormat = new Intl.DateTimeFormat('en-GB', {
  year: 'numeric',
  month: 'numeric',
  day: 'numeric',
  hour: 'numeric',
  minute: 'numeric',
});
const sameDayTimeFormat = new Intl.DateTimeFormat('en-GB', {
  hour: 'numeric',
  minute: 'numeric',
});

const AnnotationHistoryEntry: FC<{ elementHistoryPoint: ElementHistoryPoint }> = ({
  elementHistoryPoint,
}) => {
  const { projectName, elementId: currentElementId } = useParams();

  const date = elementHistoryPoint.time ? new Date(elementHistoryPoint.time) : undefined;
  const now = new Date();
  const sameDay = date
    ? now.getDate() === date.getDate() &&
      now.getMonth() === date.getMonth() &&
      now.getFullYear() === date.getFullYear()
    : undefined;

  return (
    <Link
      className={classNames(
        'history-element',
        elementHistoryPoint.element_id === currentElementId && 'selected',
      )}
      to={`/projects/${projectName}/tag/${elementHistoryPoint.element_id}`}
    >
      <p>{truncate(elementHistoryPoint.element_text, { length: 100 })}</p>
      <div className="d-flex gap-1 flex-wrap position-relative w-100">
        {elementHistoryPoint.label !== undefined && (
          <span className="badge d-flex align-center gap-1">
            {elementHistoryPoint.label ? (
              <>
                <AnnotationIcon className="flex-shrink-0" />{' '}
                <MiddleEllipsis label={elementHistoryPoint.label} />
              </>
            ) : (
              <NoAnnotationIcon />
            )}
          </span>
        )}
        <span className="badge ">id: {elementHistoryPoint.element_id}</span>
        {date && (
          <span className="badge ">
            {(sameDay ? sameDayTimeFormat : dateTimeFormat).format(date)}
          </span>
        )}
      </div>
    </Link>
  );
};

export const AnnotationHistoryList: FC = () => {
  const { appContext } = useAppContext();
  const { history, phase, currentProject } = appContext;

  const { clearAnnotationSessionHistory } = useAnnotationSessionHistory();

  return (
    <div className="horizontal center flex-column">
      <div className="d-flex justify-content-start gap-4 w-100 mb-4 align-items-center">
        <h4 className="m-0">Annotation history (last 100)</h4>
        <button
          className="btn-secondary-action d-flex align-items-center gap-2"
          onClick={() => {
            clearAnnotationSessionHistory();
          }}
        >
          <FaRegTrashAlt /> clear history
        </button>
      </div>
      <div className="annotation-history">
        {history
          .slice(0, 100)
          .filter(
            (hp) => hp.dataset === phase && hp.project_slug === currentProject?.params.project_slug,
          )
          .map((historyPoint, i) => {
            return (
              <AnnotationHistoryEntry
                key={`${historyPoint.element_id}-${i}`}
                elementHistoryPoint={historyPoint}
              />
            );
          })}
      </div>
    </div>
  );
};

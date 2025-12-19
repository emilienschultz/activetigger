import classNames from 'classnames';
import { min, reverse, truncate } from 'lodash';
import { FC, useEffect, useState } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { Link, useParams } from 'react-router-dom';
import { useGetElementById } from '../../core/api';
import { useAppContext } from '../../core/context';
import { ElementOutModel } from '../../types';

const AnnotationHistoryEntry: FC<{ elementId: string; indexInDuplicates: number }> = ({
  elementId,
  indexInDuplicates,
}) => {
  const { projectName, elementId: currentElementId } = useParams();
  const {
    appContext: { phase },
  } = useAppContext();

  const { getElementById } = useGetElementById();

  const [element, setElement] = useState<ElementOutModel | null | undefined>(null);

  useEffect(() => {
    console.log('fetch ', elementId);
    getElementById(elementId, phase).then((element) => setElement(element));
  }, [indexInDuplicates, elementId, getElementById, setElement, phase]);

  console.log(elementId, indexInDuplicates, element?.history);

  return (
    <Link
      className={classNames('history-element', elementId === currentElementId && 'selected')}
      to={`/projects/${projectName}/tag/${elementId}`}
    >
      <p>{truncate(element?.text, { length: 100 })}</p>
      <span className="badge text-truncate">
        {element?.history !== undefined
          ? (element.history as string[][])[indexInDuplicates][0]
          : null}
      </span>
    </Link>
  );
};

export const AnnotationHistoryList: FC = () => {
  const { appContext, setAppContext } = useAppContext();

  const { history } = appContext;

  return (
    <div className="horizontal center flex-column">
      <div className="d-flex justify-content-start gap-4 w-100 mb-4 align-items-center">
        <h4 className="m-0">Annotation history</h4>
        <button
          className="btn-secondary-action d-flex align-items-center gap-2"
          onClick={() => {
            setAppContext((prev) => ({ ...prev, history: [] }));
          }}
        >
          <FaRegTrashAlt /> clear history
        </button>
      </div>
      <div className="annotation-history">
        {reverse(
          history.map((previousElementId, i) => {
            // count how many time this element has been seen since this historic element has been added
            const indexInDuplicates = history
              .slice(min([i + 1, history.length]))
              .filter((h) => h === previousElementId).length;
            console.log(
              previousElementId,
              history.slice(min([i + 1, history.length])),
              indexInDuplicates,
            );
            return (
              <AnnotationHistoryEntry
                key={`${previousElementId}-${i}`}
                elementId={previousElementId}
                indexInDuplicates={indexInDuplicates}
              />
            );
          }),
        )}
      </div>
    </div>
  );
};
